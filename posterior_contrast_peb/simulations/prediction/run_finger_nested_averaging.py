"""Nested target-focused predictive averaging for the empirical FingerData summaries."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import norm


HERE = Path(__file__).resolve().parent
CLOSURE = HERE.parent
EMPIRICAL = CLOSURE / "posterior_anchored"
for directory in (CLOSURE, EMPIRICAL):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from run_empirical_anchor import (  # noqa: E402
    fit_candidate,
    load_anchor,
    target_normalizer,
)
from prediction_models import (  # noqa: E402
    CANDIDATES,
    group_target_moments,
    pointwise_log_densities,
    predictive_parameters,
)
from specification_averaging import (  # noqa: E402
    gaussian_mixture_marginal_interval,
    gaussian_mixture_moments,
    mixture_log_density,
    pseudo_bma_plus_weights,
    regularized_stacking_weights,
)
_DATA: dict[str, object] = {}


def initialize_worker(data: dict[str, object]) -> None:
    global _DATA
    _DATA = data


def stratified_folds(indices: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(folds)]
    for group in np.unique(groups[indices]):
        members = indices[groups[indices] == group].copy()
        rng.shuffle(members)
        for position, subject in enumerate(members):
            buckets[position % folds].append(int(subject))
    return [np.asarray(sorted(bucket), dtype=int) for bucket in buckets if bucket]


def stratified_bootstrap_indices(
    rng: np.random.Generator, groups: np.ndarray
) -> np.ndarray:
    """Resample subjects within each diagnostic group, preserving group sizes."""

    groups = np.asarray(groups)
    group_indices = [np.flatnonzero(groups == group) for group in np.unique(groups)]
    return np.concatenate(
        [
            rng.choice(group_members, size=len(group_members), replace=True)
            for group_members in group_indices
        ]
    )


def candidate_predictions(
    training: np.ndarray,
    evaluation: np.ndarray,
    contrast: np.ndarray,
) -> tuple[np.ndarray, list[tuple[object, bool]]]:
    means = np.asarray(_DATA["means"])
    covariances = np.asarray(_DATA["covariances"])
    design = np.asarray(_DATA["design"])
    qbase = np.asarray(_DATA["qbase"])
    values = means[evaluation] @ contrast.T
    parameters = []
    fits = []
    for candidate in CANDIDATES:
        fit, target_space = fit_candidate(
            means[training],
            design[training],
            covariances[training],
            qbase,
            contrast,
            candidate,
        )
        fits.append((fit, target_space))
        parameters.append(
            predictive_parameters(
                fit,
                target_space,
                design[evaluation],
                covariances[evaluation],
                contrast,
                means.shape[1],
            )
        )
    return pointwise_log_densities(values, parameters), fits


def full_data_loso_fold(held_out: int) -> tuple[int, np.ndarray]:
    """Return candidate log densities for one full-data LOSO fold."""
    n_subjects = len(np.asarray(_DATA["means"]))
    target_dimension = int(_DATA["target_dimension"])
    contrast = np.asarray(_DATA["contrasts"][target_dimension])
    training = np.delete(np.arange(n_subjects), held_out)
    densities, _ = candidate_predictions(
        training, np.asarray([held_out]), contrast
    )
    return held_out, densities[0]


def outer_fold(held_out: int) -> list[dict[str, object]]:
    n_subjects = len(np.asarray(_DATA["means"]))
    groups = np.asarray(_DATA["groups"])
    target_dimension = int(_DATA["target_dimension"])
    contrast = np.asarray(_DATA["contrasts"][target_dimension])
    outer_training = np.delete(np.arange(n_subjects), held_out)
    inner_folds = stratified_folds(
        outer_training, groups, int(_DATA["inner_folds"]), int(_DATA["seed"]) + held_out
    )
    inner_rows = np.empty((len(outer_training), len(CANDIDATES)))
    outer_position = {subject: position for position, subject in enumerate(outer_training)}
    for inner_validation in inner_folds:
        inner_training = np.setdiff1d(outer_training, inner_validation, assume_unique=True)
        densities, _ = candidate_predictions(inner_training, inner_validation, contrast)
        for row, subject in enumerate(inner_validation):
            inner_rows[outer_position[int(subject)]] = densities[row]
    stacking = regularized_stacking_weights(
        inner_rows, CANDIDATES, entropy_strength=float(_DATA["entropy_strength"])
    )
    pseudo = pseudo_bma_plus_weights(
        inner_rows,
        CANDIDATES,
        bootstrap_repetitions=int(_DATA["pseudo_repetitions"]),
        seed=int(_DATA["seed"]) + 10000 + held_out,
    )
    hard = np.zeros(len(CANDIDATES))
    hard[int(np.argmax(inner_rows.sum(axis=0)))] = 1.0
    equal = np.full(len(CANDIDATES), 1.0 / len(CANDIDATES))
    held_density, _ = candidate_predictions(
        outer_training, np.asarray([held_out]), contrast
    )
    methods = {
        "hard_inner_selection": hard,
        "equal_average": equal,
        "regularized_stacking": stacking.weights,
        "pseudo_bma_plus": pseudo.weights,
    }
    rows = []
    for method, weights in methods.items():
        row = {
            "target_dimension": target_dimension,
            "held_out": held_out,
            "method": method,
            "log_predictive_density": float(mixture_log_density(held_density, weights)[0]),
            "weight_source_single": float(weights[0]),
            "weight_source_all": float(weights[1]),
            "weight_target_direct": float(weights[2]),
            "weight_rotated_fitted": float(weights[3]),
            "weight_fit_converged": int(
                stacking.converged if method == "regularized_stacking" else True
            ),
        }
        rows.append(row)
    for index, candidate in enumerate(CANDIDATES):
        rows.append(
            {
                "target_dimension": target_dimension,
                "held_out": held_out,
                "method": candidate,
                "log_predictive_density": float(held_density[0, index]),
                "weight_source_single": float(index == 0),
                "weight_source_all": float(index == 1),
                "weight_target_direct": float(index == 2),
                "weight_rotated_fitted": float(index == 3),
                "weight_fit_converged": 1,
            }
        )
    return rows


def full_data_summary(data: dict[str, object], target_dimension: int, args) -> dict[str, object]:
    worker_data = {**data, "target_dimension": target_dimension}
    initialize_worker(worker_data)
    n_subjects = len(np.asarray(data["means"]))
    contrast = np.asarray(data["contrasts"][target_dimension])
    pointwise = np.full((n_subjects, len(CANDIDATES)), np.nan)
    checkpoint_path = HERE / f"{args.tag}_R{target_dimension}_full_loso_checkpoint.csv"
    if args.resume and checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path).set_index("held_out")
        for held_out, row in checkpoint.iterrows():
            pointwise[int(held_out)] = row.loc[list(CANDIDATES)].to_numpy(float)
    pending = np.flatnonzero(~np.isfinite(pointwise).all(axis=1)).tolist()
    if pending:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(worker_data,),
        ) as executor:
            for completed, (held_out, densities) in enumerate(
                executor.map(full_data_loso_fold, pending, chunksize=1), start=1
            ):
                pointwise[held_out] = densities
                if completed % args.checkpoint_every == 0 or completed == len(pending):
                    completed_rows = np.flatnonzero(np.isfinite(pointwise).all(axis=1))
                    checkpoint = pd.DataFrame(
                        pointwise[completed_rows], columns=CANDIDATES
                    )
                    checkpoint.insert(0, "held_out", completed_rows)
                    temporary = checkpoint_path.with_suffix(
                        checkpoint_path.suffix + ".tmp"
                    )
                    checkpoint.to_csv(temporary, index=False)
                    temporary.replace(checkpoint_path)
                    print(
                        f"R={target_dimension}: completed full-data LOSO "
                        f"{completed}/{len(pending)} pending",
                        flush=True,
                    )
    if not np.isfinite(pointwise).all():
        raise RuntimeError(f"Incomplete full-data LOSO scores for R={target_dimension}")
    stacking = regularized_stacking_weights(
        pointwise, CANDIDATES, entropy_strength=args.entropy_strength
    )
    pseudo = pseudo_bma_plus_weights(
        pointwise,
        CANDIDATES,
        bootstrap_repetitions=args.pseudo_repetitions,
        seed=args.seed + target_dimension,
    )
    indices = np.arange(n_subjects)
    _, fits = candidate_predictions(indices, np.asarray([0]), contrast)
    normalizer = target_normalizer(np.asarray(data["qbase"]), contrast)
    candidate_means = []
    candidate_covariances = []
    normalized_candidate_means = []
    normalized_candidate_covariances = []
    for fit, target_space in fits:
        mean, covariance = group_target_moments(
            fit,
            target_space,
            contrast,
            np.eye(target_dimension),
            np.asarray(data["means"]).shape[1],
        )
        candidate_means.append(mean)
        candidate_covariances.append(covariance)
        normalized_mean, normalized_covariance = group_target_moments(
            fit,
            target_space,
            contrast,
            normalizer,
            np.asarray(data["means"]).shape[1],
        )
        normalized_candidate_means.append(normalized_mean)
        normalized_candidate_covariances.append(normalized_covariance)
    candidate_means = np.asarray(candidate_means)
    candidate_covariances = np.asarray(candidate_covariances)
    normalized_candidate_means = np.asarray(normalized_candidate_means)
    normalized_candidate_covariances = np.asarray(normalized_candidate_covariances)
    mixture_mean, mixture_covariance = gaussian_mixture_moments(
        candidate_means, candidate_covariances, stacking.weights
    )
    lower, upper = gaussian_mixture_marginal_interval(
        candidate_means, candidate_covariances, stacking.weights
    )
    component_sd = np.sqrt(
        np.maximum(np.diagonal(candidate_covariances, axis1=1, axis2=2), 1e-15)
    )
    directional_support = np.sum(
        stacking.weights[:, None] * norm.cdf(candidate_means / component_sd), axis=0
    )
    normalized_mixture_mean, normalized_mixture_covariance = gaussian_mixture_moments(
        normalized_candidate_means,
        normalized_candidate_covariances,
        stacking.weights,
    )
    rng = np.random.default_rng(args.seed + 50000 + target_dimension)
    groups = np.asarray(data["groups"])
    bootstrap_weights = []
    for _ in range(args.weight_bootstrap_reps):
        sampled = stratified_bootstrap_indices(rng, groups)
        result = regularized_stacking_weights(
            pointwise[sampled], CANDIDATES, entropy_strength=args.entropy_strength
        )
        bootstrap_weights.append(result.weights)
    bootstrap_weights = np.asarray(bootstrap_weights)
    weight_interval = {
        candidate: {
            "mean": float(stacking.weights[index]),
            "lower": float(np.quantile(bootstrap_weights[:, index], 0.025)),
            "upper": float(np.quantile(bootstrap_weights[:, index], 0.975)),
        }
        for index, candidate in enumerate(CANDIDATES)
    }
    return {
        "target_dimension": target_dimension,
        "candidate_names": CANDIDATES,
        "stacking_weights": weight_interval,
        "pseudo_bma_plus_weights": {
            candidate: float(pseudo.weights[index])
            for index, candidate in enumerate(CANDIDATES)
        },
        "effective_candidate_count": stacking.effective_count,
        "candidate_group_means": candidate_means.tolist(),
        "candidate_group_covariances": candidate_covariances.tolist(),
        "averaged_group_mean_original_scale": mixture_mean.tolist(),
        "averaged_group_covariance_original_scale": mixture_covariance.tolist(),
        "averaged_equal_tail_lower_original_scale": lower.tolist(),
        "averaged_equal_tail_upper_original_scale": upper.tolist(),
        "averaged_directional_support_original_scale": directional_support.tolist(),
        "averaged_group_mean_qbase_standardized": normalized_mixture_mean.tolist(),
        "averaged_group_covariance_qbase_standardized": normalized_mixture_covariance.tolist(),
        "weight_interpretation": "predictive combination weights, not posterior model probabilities",
        "weight_bootstrap": "subject-level bootstrap stratified by diagnostic group",
    }


def summarize_nested(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (target_dimension, method), group in rows.groupby(["target_dimension", "method"]):
        summaries.append(
            {
                "target_dimension": target_dimension,
                "method": method,
                "elpd": float(group["log_predictive_density"].sum()),
                "mean_log_score": float(group["log_predictive_density"].mean()),
                "se_elpd": math.sqrt(
                    len(group) * float(group["log_predictive_density"].var(ddof=1))
                ),
                "mean_weight_source_single": float(group["weight_source_single"].mean()),
                "mean_weight_source_all": float(group["weight_source_all"].mean()),
                "mean_weight_target_direct": float(group["weight_target_direct"].mean()),
                "mean_weight_rotated_fitted": float(group["weight_rotated_fitted"].mean()),
                "weight_convergence_rate": float(group["weight_fit_converged"].mean()),
            }
        )
    return pd.DataFrame(summaries)


def paired_nested(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for target_dimension, subset in rows.groupby("target_dimension"):
        wide = subset.pivot(
            index="held_out", columns="method", values="log_predictive_density"
        )
        methods = tuple(wide.columns)
        for first_index, first in enumerate(methods):
            for second in methods[first_index + 1 :]:
                difference = wide[first] - wide[second]
                output.append(
                    {
                        "target_dimension": target_dimension,
                        "first": first,
                        "second": second,
                        "delta_elpd_first_minus_second": float(difference.sum()),
                        "paired_se": math.sqrt(
                            len(difference) * float(difference.var(ddof=1))
                        ),
                    }
                )
    return pd.DataFrame(output)


def write_checkpoint(rows: list[dict[str, object]], path: Path) -> None:
    """Atomically persist completed outer folds for interruption-safe resumption."""
    frame = pd.DataFrame(rows).sort_values(
        ["target_dimension", "held_out", "method"]
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--covariates", type=Path, required=True)
    parser.add_argument("--target-dims", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--inner-folds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--entropy-strength", type=float, default=1.0)
    parser.add_argument("--pseudo-repetitions", type=int, default=500)
    parser.add_argument("--weight-bootstrap-reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--tag", default="finger_nested_averaging_v1")
    parser.add_argument("--max-outer", type=int)
    parser.add_argument("--skip-full-summary", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    anchor = load_anchor(args.input, args.covariates)
    worker_base = {
        key: anchor[key]
        for key in (
            "means",
            "covariances",
            "qbase",
            "design",
            "contrasts",
            "subjects",
            "groups",
        )
    }
    worker_base.update(
        {
            "inner_folds": args.inner_folds,
            "entropy_strength": args.entropy_strength,
            "pseudo_repetitions": args.pseudo_repetitions,
            "seed": args.seed,
        }
    )
    checkpoint_path = HERE / f"{args.tag}_nested_checkpoint.csv"
    rows: list[dict[str, object]] = []
    completed_keys: set[tuple[int, int]] = set()
    if args.resume and checkpoint_path.exists():
        previous = pd.read_csv(checkpoint_path)
        rows = previous.to_dict("records")
        counts = previous.groupby(["target_dimension", "held_out"])["method"].nunique()
        completed_keys = {
            (int(target_dimension), int(held_out))
            for (target_dimension, held_out), count in counts.items()
            if int(count) == len(CANDIDATES) + 4
        }
        print(
            f"Resuming from {len(completed_keys)} complete outer folds",
            flush=True,
        )
    outer_subjects = list(range(len(anchor["subjects"])))
    if args.max_outer is not None:
        outer_subjects = outer_subjects[: args.max_outer]
    for target_dimension in args.target_dims:
        pending_subjects = [
            held_out
            for held_out in outer_subjects
            if (target_dimension, held_out) not in completed_keys
        ]
        if not pending_subjects:
            print(f"R={target_dimension}: all outer folds already complete", flush=True)
            continue
        worker_data = {**worker_base, "target_dimension": target_dimension}
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=initialize_worker,
            initargs=(worker_data,),
        ) as executor:
            for completed, fold_rows in enumerate(
                executor.map(outer_fold, pending_subjects, chunksize=1), start=1
            ):
                rows.extend(fold_rows)
                if (
                    completed % args.checkpoint_every == 0
                    or completed == len(pending_subjects)
                ):
                    write_checkpoint(rows, checkpoint_path)
                    print(
                        f"R={target_dimension}: completed outer folds "
                        f"{completed}/{len(pending_subjects)} pending "
                        f"({len(pending_subjects)} total pending at start)",
                        flush=True,
                    )
    pointwise = pd.DataFrame(rows)
    expected_folds = len(outer_subjects) * len(args.target_dims)
    actual_folds = pointwise[["target_dimension", "held_out"]].drop_duplicates().shape[0]
    if actual_folds != expected_folds:
        raise RuntimeError(
            f"Incomplete nested result: expected {expected_folds} outer folds, "
            f"found {actual_folds}"
        )
    pointwise.to_csv(HERE / f"{args.tag}_nested_pointwise.csv", index=False)
    summary = summarize_nested(pointwise)
    summary.to_csv(HERE / f"{args.tag}_nested_summary.csv", index=False)
    paired_nested(pointwise).to_csv(
        HERE / f"{args.tag}_nested_pairwise.csv", index=False
    )
    if not args.skip_full_summary:
        final = [full_data_summary(worker_base, dimension, args) for dimension in args.target_dims]
        (HERE / f"{args.tag}_full_data_averaged_posterior.json").write_text(
            json.dumps(final, indent=2), encoding="utf-8"
        )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
