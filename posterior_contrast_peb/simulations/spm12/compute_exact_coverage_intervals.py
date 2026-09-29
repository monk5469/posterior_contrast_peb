#!/usr/bin/env python3
"""Replace Wald coverage intervals with finite-sample exact intervals.

The released SPM12 summary tables store, for every cell, the replication
count, target dimension, mean coverage and its across-replication standard
error. For R=1, each replication contributes a Bernoulli indicator. For R=2,
each replication contributes the paired mean K/2, where K is in {0, 1, 2}.
The outcome counts are recovered from the released mean and standard error.

R=1 intervals use central Clopper-Pearson inversion. R=2 intervals invert
one-sided exact trinomial tests while maximising the tail probability over
the nuisance allocation compatible with each candidate mean.
"""

from __future__ import annotations

import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import beta


@dataclass(frozen=True)
class ExactInterval:
    lower: float
    upper: float
    method: str
    counts: tuple[int, ...]


def clopper_pearson(successes: int, n: int, alpha: float) -> ExactInterval:
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2, successes, n - successes + 1)
    )
    upper = 1.0 if successes == n else float(
        beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    )
    return ExactInterval(
        lower,
        upper,
        "central_exact_binomial_clopper_pearson",
        (n - successes, successes),
    )


def trinomial_sample_space(n: int) -> tuple[np.ndarray, ...]:
    counts = [
        (n0, n1, n - n0 - n1)
        for n0 in range(n + 1)
        for n1 in range(n - n0 + 1)
    ]
    n0, n1, n2 = (np.asarray(values, dtype=float) for values in zip(*counts))
    score = (n1 + 2 * n2) / (2 * n)
    log_coefficient = (
        gammaln(n + 1) - gammaln(n0 + 1) - gammaln(n1 + 1) - gammaln(n2 + 1)
    )
    return n0, n1, n2, score, log_coefficient


def tail_probability(
    q: float,
    mu: float,
    observed: float,
    side: str,
    space: tuple[np.ndarray, ...],
) -> float:
    n0, n1, n2, score, log_coefficient = space
    probabilities = np.asarray([1 - 2 * mu + q, 2 * (mu - q), q])
    if np.min(probabilities) < -1e-12:
        return -math.inf
    probabilities = np.clip(probabilities, 0.0, 1.0)
    log_probability = (
        log_coefficient
        + xlogy(n0, probabilities[0])
        + xlogy(n1, probabilities[1])
        + xlogy(n2, probabilities[2])
    )
    if side == "upper":
        mask = score >= observed - 1e-15
    elif side == "lower":
        mask = score <= observed + 1e-15
    else:
        raise ValueError(f"Unknown tail side: {side}")
    return float(np.clip(np.exp(logsumexp(log_probability[mask])), 0.0, 1.0))


def max_tail_at_mean(
    mu: float,
    observed: float,
    side: str,
    space: tuple[np.ndarray, ...],
    grid_size: int,
) -> float:
    q_lower = max(0.0, 2 * mu - 1)
    q_upper = mu
    if q_upper - q_lower <= 1e-14:
        return tail_probability(q_lower, mu, observed, side, space)

    grid = np.linspace(q_lower, q_upper, grid_size)
    values = np.asarray([
        tail_probability(float(q), mu, observed, side, space) for q in grid
    ])
    candidate_indices = {0, grid_size - 1, int(np.argmax(values))}
    candidate_indices.update(np.argsort(values)[-8:].tolist())
    best = float(np.max(values))
    for index in sorted(candidate_indices):
        left = float(grid[max(0, index - 1)])
        right = float(grid[min(grid_size - 1, index + 1)])
        if right - left <= 1e-14:
            continue
        result = minimize_scalar(
            lambda candidate: -tail_probability(
                candidate, mu, observed, side, space
            ),
            bounds=(left, right),
            method="bounded",
            options={"xatol": 1e-13, "maxiter": 500},
        )
        best = max(best, -float(result.fun))
    return min(max(best, 0.0), 1.0)


def invert_one_sided(
    observed: float,
    alpha_tail: float,
    bound: str,
    space: tuple[np.ndarray, ...],
    grid_size: int,
    root_tolerance: float,
) -> float:
    if bound == "lower":
        if observed <= 0:
            return 0.0
        low, high, side = 0.0, observed, "upper"
        for _ in range(80):
            middle = (low + high) / 2
            if max_tail_at_mean(middle, observed, side, space, grid_size) >= alpha_tail:
                high = middle
            else:
                low = middle
            if high - low <= root_tolerance:
                break
        return high
    if bound == "upper":
        if observed >= 1:
            return 1.0
        low, high, side = observed, 1.0, "lower"
        for _ in range(80):
            middle = (low + high) / 2
            if max_tail_at_mean(middle, observed, side, space, grid_size) >= alpha_tail:
                low = middle
            else:
                high = middle
            if high - low <= root_tolerance:
                break
        return low
    raise ValueError(f"Unknown bound: {bound}")


def exact_trinomial(
    counts: tuple[int, int, int],
    alpha: float,
    grid_size: int,
    root_tolerance: float,
) -> ExactInterval:
    n = sum(counts)
    observed = (counts[1] + 2 * counts[2]) / (2 * n)
    space = trinomial_sample_space(n)
    lower = invert_one_sided(
        observed, alpha / 2, "lower", space, grid_size, root_tolerance
    )
    upper = invert_one_sided(
        observed, alpha / 2, "upper", space, grid_size, root_tolerance
    )
    return ExactInterval(
        lower,
        upper,
        "central_exact_trinomial_test_inversion",
        counts,
    )


def compute_interval_task(
    task: tuple[tuple[int, int, tuple[int, ...]], float, int, float]
) -> tuple[tuple[int, int, tuple[int, ...]], ExactInterval]:
    key, alpha, grid_size, root_tolerance = task
    n, target_dimension, counts = key
    if target_dimension == 1:
        interval = clopper_pearson(counts[1], n, alpha)
    else:
        interval = exact_trinomial(counts, alpha, grid_size, root_tolerance)
    return key, interval


def infer_counts(n: int, target_dimension: int, mean: float, se: float) -> tuple[int, ...]:
    if target_dimension == 1:
        successes = int(round(n * mean))
        if not math.isclose(successes / n, mean, abs_tol=1e-10):
            raise ValueError(f"Cannot recover binary count from n={n}, mean={mean}")
        counts = (n - successes, successes)
        recovered = np.repeat((0.0, 1.0), counts)
        recovered_se = float(np.std(recovered, ddof=1) / math.sqrt(n))
        if not math.isclose(recovered_se, se, abs_tol=1e-10):
            raise ValueError(f"SE does not recover: {se} versus {recovered_se}")
        return counts

    if target_dimension != 2:
        raise ValueError(f"Unsupported target dimension: {target_dimension}")
    sample_variance = n * se * se
    sum_values = n * mean
    sum_squares = (n - 1) * sample_variance + n * mean * mean
    n1 = int(round(4 * (sum_values - sum_squares)))
    n2 = int(round(sum_values - n1 / 2))
    n0 = n - n1 - n2
    counts = (n0, n1, n2)
    if min(counts) < 0:
        raise ValueError(f"Negative recovered count: {counts}")
    recovered = np.repeat((0.0, 0.5, 1.0), counts)
    recovered_mean = float(np.mean(recovered))
    recovered_se = float(np.std(recovered, ddof=1) / math.sqrt(n))
    if not math.isclose(recovered_mean, mean, abs_tol=1e-10):
        raise ValueError(f"Mean does not recover: {mean} versus {recovered_mean}")
    if not math.isclose(recovered_se, se, abs_tol=1e-10):
        raise ValueError(f"SE does not recover: {se} versus {recovered_se}")
    return counts


def coverage_prefixes(columns: list[str]) -> list[str]:
    prefixes = []
    for column in columns:
        if not column.endswith("_coverage_mean"):
            continue
        prefix = column[:-5]
        required = {
            f"{prefix}_mean",
            f"{prefix}_se",
            f"{prefix}_lower95",
            f"{prefix}_upper95",
        }
        if required.issubset(columns):
            prefixes.append(prefix)
    return prefixes


def update_table(
    table: pd.DataFrame,
    alpha: float,
    grid_size: int,
    root_tolerance: float,
    workers: int,
) -> pd.DataFrame:
    output = table.copy()
    prefixes = coverage_prefixes(list(output.columns))
    if not prefixes:
        raise ValueError("No complete coverage families found")

    row_keys: dict[tuple[int, str], tuple[int, int, tuple[int, ...]]] = {}
    unique_keys: set[tuple[int, int, tuple[int, ...]]] = set()
    for prefix in prefixes:
        for index, row in output.iterrows():
            mean = row[f"{prefix}_mean"]
            se = row[f"{prefix}_se"]
            if pd.isna(mean) or pd.isna(se):
                continue
            n = int(row["replications"])
            target_dimension = int(row["target_dimension"])
            counts = infer_counts(n, target_dimension, float(mean), float(se))
            key = (n, target_dimension, counts)
            row_keys[(index, prefix)] = key
            unique_keys.add(key)

    tasks = [
        (key, alpha, grid_size, root_tolerance) for key in sorted(unique_keys)
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        interval_cache = dict(executor.map(compute_interval_task, tasks))

    for prefix in prefixes:
        method_column = f"{prefix}_interval_method"
        counts_column = f"{prefix}_outcome_counts"
        output[method_column] = ""
        output[counts_column] = ""
        for index, row in output.iterrows():
            if (index, prefix) not in row_keys:
                continue
            interval = interval_cache[row_keys[(index, prefix)]]
            output.at[index, f"{prefix}_lower95"] = interval.lower
            output.at[index, f"{prefix}_upper95"] = interval.upper
            output.at[index, method_column] = interval.method
            output.at[index, counts_column] = ",".join(map(str, interval.counts))

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=4097)
    parser.add_argument("--root-tolerance", type=float, default=1e-10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.input)
    updated = update_table(
        table, args.alpha, args.grid_size, args.root_tolerance, args.workers
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(args.output, index=False)
    print(f"Updated {len(updated)} rows and wrote {args.output}")


if __name__ == "__main__":
    main()
