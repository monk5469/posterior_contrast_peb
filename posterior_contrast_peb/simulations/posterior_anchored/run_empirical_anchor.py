"""Empirically anchored covariance-specification recovery experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
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
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ANALYSIS = HERE.parents[1] / "analysis"
for directory in (PARENT, ANALYSIS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from covariance_specification import (  # noqa: E402
    fit_heteroscedastic_reml,
    fixed_effect_prediction_covariance,
    multivariate_normal_logpdf,
    source_covariance_components,
    target_direction_diagnostic,
    target_geometry,
)


WORKFLOWS = (
    "source_single",
    "source_all",
    "target_direct",
    "rotated_fitted",
)
_DATA: dict[str, object] = {}


def zscore(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(float)
    return (array - array.mean()) / array.std(ddof=0)


def bilateral_target_matrix(parameter_names: list[str]) -> np.ndarray:
    """Return the left/right FingerData target basis used in the manuscript."""

    name_to_index = {name: index for index, name in enumerate(parameter_names)}
    contrast = np.zeros((2, len(parameter_names)))
    for row, target in enumerate((1, 2)):
        contrast[row, name_to_index[f"B({target},3,3)"]] = 1.0
        contrast[row, name_to_index[f"B({target},3,1)"]] = -0.5
        contrast[row, name_to_index[f"B({target},3,2)"]] = -0.5
    return contrast


@dataclass(frozen=True)
class Cell:
    target_dimension: int
    kappa: float
    target_variance: float
    observation_scale: float
    replication: int
    seed: int
    test_subjects: int


def _sym(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _inverse_sqrt(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    values, vectors = np.linalg.eigh(_sym(matrix))
    cutoff = tolerance * max(float(values.max(initial=0.0)), 1.0)
    if np.any(values < -cutoff):
        raise ValueError("Matrix is not positive semidefinite.")
    keep = values > cutoff
    if not np.any(keep):
        raise ValueError("Matrix has zero numerical rank.")
    return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T


def _psd_cholesky(matrix: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    values, vectors = np.linalg.eigh(_sym(matrix))
    floor = tolerance * max(float(values.max(initial=0.0)), 1.0)
    if float(values.min(initial=0.0)) < -100.0 * floor:
        raise ValueError("Covariance has a materially negative eigenvalue.")
    return vectors @ np.diag(np.sqrt(np.maximum(values, floor)))


def build_design(
    subjects: list[str], groups: list[str], covariate_file: Path
) -> tuple[np.ndarray, list[str]]:
    covariates = pd.read_csv(covariate_file)
    required = {"subject", "age", "sex", "mean_fd", "pct_fd_gt_0p5"}
    missing = required.difference(covariates.columns)
    if missing:
        raise ValueError(f"Covariate file is missing columns: {sorted(missing)}")
    covariates = covariates[list(required)]
    metadata = pd.DataFrame({"subject": subjects, "group": groups})
    metadata = metadata.merge(covariates, on="subject", how="left")
    if metadata[["age", "sex", "mean_fd", "pct_fd_gt_0p5"]].isna().any().any():
        raise ValueError("FingerData covariates are incomplete.")
    metadata["intercept"] = 1.0
    metadata["CD_minus_HC"] = metadata["group"].str.upper().eq("CD").astype(float)
    metadata["ULD_minus_HC"] = metadata["group"].str.upper().eq("ULD").astype(float)
    metadata["age_z"] = zscore(metadata["age"])
    metadata["male"] = metadata["sex"].str.upper().eq("M").astype(float)
    metadata["mean_fd_z"] = zscore(metadata["mean_fd"])
    metadata["spike_z"] = zscore(metadata["pct_fd_gt_0p5"])
    names = [
        "intercept",
        "CD_minus_HC",
        "ULD_minus_HC",
        "age_z",
        "male",
        "mean_fd_z",
        "spike_z",
    ]
    design = metadata[names].to_numpy(float)
    if np.linalg.matrix_rank(design) != len(names):
        raise ValueError("FingerData design is rank deficient.")
    return design, names


def load_anchor(input_file: Path, covariate_file: Path) -> dict[str, object]:
    source = loadmat(input_file, squeeze_me=True)
    means = np.asarray(source["mu"], dtype=float)
    covariances = np.moveaxis(np.asarray(source["C"], dtype=float), 2, 0)
    qbase = _sym(np.asarray(source["source_single_qbase"], dtype=float))
    parameter_names = [str(value) for value in np.atleast_1d(source["Pnames"])]
    subjects = [str(value) for value in np.atleast_1d(source["subject"])]
    groups = [str(value) for value in np.atleast_1d(source["group"])]
    scalar = np.atleast_2d(np.asarray(source["L"], dtype=float))
    bilateral = bilateral_target_matrix(parameter_names)
    if not np.allclose(scalar, 0.5 * bilateral.sum(axis=0, keepdims=True)):
        raise ValueError("Scalar and bilateral FingerData targets are inconsistent.")
    design, design_names = build_design(subjects, groups, covariate_file)
    stabilized = np.stack([_sym(covariance) for covariance in covariances])
    cholesky = np.stack([_psd_cholesky(covariance) for covariance in stabilized])
    fixed_coefficients = np.linalg.pinv(design) @ means
    return {
        "means": means,
        "covariances": stabilized,
        "covariance_cholesky": cholesky,
        "qbase": qbase,
        "design": design,
        "design_names": design_names,
        "fixed_coefficients": fixed_coefficients,
        "contrasts": {1: scalar, 2: bilateral},
        "subjects": subjects,
        "groups": groups,
        "parameter_names": parameter_names,
    }


def true_random_covariance(
    qbase: np.ndarray,
    contrast: np.ndarray,
    target_variance: float,
    kappa: float,
) -> np.ndarray:
    geometry = target_geometry(qbase, contrast)
    target_projector = geometry.target_basis.T @ geometry.target_basis
    nuisance_projector = geometry.nuisance_basis.T @ geometry.nuisance_basis
    whitened = target_variance * target_projector + (
        kappa * target_variance
    ) * nuisance_projector
    return _sym(geometry.colouring @ whitened @ geometry.colouring.T)


def projected_scales(
    covariance: np.ndarray,
    qbase: np.ndarray,
    contrast: np.ndarray,
) -> tuple[float, float]:
    geometry = target_geometry(qbase, contrast)
    whitened = _sym(geometry.whitening @ covariance @ geometry.whitening.T)
    target = float(
        np.trace(geometry.target_basis @ whitened @ geometry.target_basis.T)
    ) / geometry.target_dimension
    nuisance_dimension = geometry.nuisance_basis.shape[0]
    nuisance = float(
        np.trace(geometry.nuisance_basis @ whitened @ geometry.nuisance_basis.T)
    ) / nuisance_dimension
    return target, nuisance


def target_normalizer(qbase: np.ndarray, contrast: np.ndarray) -> np.ndarray:
    return _inverse_sqrt(contrast @ qbase @ contrast.T)


def initialize_worker(data: dict[str, object]) -> None:
    global _DATA
    _DATA = data


def draw_population(
    rng: np.random.Generator,
    design: np.ndarray,
    fixed_coefficients: np.ndarray,
    random_cholesky: np.ndarray,
    observation_cholesky: np.ndarray,
) -> np.ndarray:
    n_subjects, dimension = observation_cholesky.shape[:2]
    random = rng.normal(size=(n_subjects, dimension)) @ random_cholesky.T
    error = np.einsum(
        "sij,sj->si", observation_cholesky, rng.normal(size=(n_subjects, dimension))
    )
    return design @ fixed_coefficients + random + error


def fit_candidate(
    observations: np.ndarray,
    design: np.ndarray,
    observation_covariances: np.ndarray,
    qbase: np.ndarray,
    contrast: np.ndarray,
    workflow: str,
):
    if workflow == "target_direct":
        target_observations = observations @ contrast.T
        target_covariances = np.stack(
            [contrast @ covariance @ contrast.T for covariance in observation_covariances]
        )
        target_qbase = contrast @ qbase @ contrast.T
        components = source_covariance_components(
            target_qbase, np.eye(contrast.shape[0]), "target_direct"
        )
        fit = fit_heteroscedastic_reml(
            target_observations, design, target_covariances, components
        )
        return fit, True
    components = source_covariance_components(qbase, contrast, workflow)
    fit = fit_heteroscedastic_reml(
        observations, design, observation_covariances, components
    )
    return fit, False


def coefficient_coverage(
    fit,
    target_space: bool,
    true_coefficients: np.ndarray,
    contrast: np.ndarray,
    normalizer: np.ndarray,
    source_dimension: int,
    design_indices: tuple[int, ...] = (1, 2),
) -> tuple[int, int]:
    target_dimension = contrast.shape[0]
    if target_space:
        estimate = fit.coefficients
        map_target = normalizer
        fitted_dimension = target_dimension
    else:
        estimate = fit.coefficients @ contrast.T
        map_target = normalizer @ contrast
        fitted_dimension = source_dimension
    estimate = estimate @ normalizer.T
    truth = true_coefficients @ contrast.T @ normalizer.T
    covered = 0
    total = 0
    for design_index in design_indices:
        selector = np.zeros((1, true_coefficients.shape[0]))
        selector[0, design_index] = 1.0
        mapping = np.kron(selector, map_target)
        covariance = _sym(mapping @ fit.coefficient_covariance @ mapping.T)
        if covariance.shape != (target_dimension, target_dimension):
            raise RuntimeError("Unexpected coefficient covariance projection shape.")
        for target_index in range(target_dimension):
            standard_error = math.sqrt(max(float(covariance[target_index, target_index]), 0.0))
            lower = estimate[design_index, target_index] - 1.96 * standard_error
            upper = estimate[design_index, target_index] + 1.96 * standard_error
            covered += int(lower <= truth[design_index, target_index] <= upper)
            total += 1
    if fitted_dimension <= 0:
        raise RuntimeError("Invalid fitted dimension.")
    return covered, total


def score_candidate(
    fit,
    target_space: bool,
    test_observations: np.ndarray,
    test_design: np.ndarray,
    test_covariances: np.ndarray,
    contrast: np.ndarray,
) -> float:
    target_values = test_observations @ contrast.T
    scores = []
    for subject in range(len(test_observations)):
        if target_space:
            predictive_mean = test_design[subject] @ fit.coefficients
            fixed_covariance = fixed_effect_prediction_covariance(
                fit.coefficient_covariance,
                test_design[subject],
                np.eye(contrast.shape[0]),
                contrast.shape[0],
            )
            random_target = fit.random_covariance
        else:
            predictive_mean = contrast @ (test_design[subject] @ fit.coefficients)
            fixed_covariance = fixed_effect_prediction_covariance(
                fit.coefficient_covariance,
                test_design[subject],
                contrast,
                test_observations.shape[1],
            )
            random_target = contrast @ fit.random_covariance @ contrast.T
        predictive_covariance = _sym(
            contrast @ test_covariances[subject] @ contrast.T
            + random_target
            + fixed_covariance
        )
        scores.append(
            multivariate_normal_logpdf(
                target_values[subject], predictive_mean, predictive_covariance
            )
        )
    return float(np.mean(scores))


def oracle_score(
    test_observations: np.ndarray,
    test_design: np.ndarray,
    test_covariances: np.ndarray,
    fixed_coefficients: np.ndarray,
    true_random: np.ndarray,
    contrast: np.ndarray,
) -> float:
    values = test_observations @ contrast.T
    scores = []
    random_target = contrast @ true_random @ contrast.T
    for subject in range(len(test_observations)):
        mean = contrast @ (test_design[subject] @ fixed_coefficients)
        covariance = _sym(
            contrast @ test_covariances[subject] @ contrast.T + random_target
        )
        scores.append(multivariate_normal_logpdf(values[subject], mean, covariance))
    return float(np.mean(scores))


def run_cell(cell: Cell) -> tuple[list[dict[str, object]], dict[str, object]]:
    rng = np.random.default_rng(cell.seed)
    means = np.asarray(_DATA["means"])
    covariances = np.asarray(_DATA["covariances"]) * cell.observation_scale
    covariance_cholesky = (
        np.asarray(_DATA["covariance_cholesky"]) * math.sqrt(cell.observation_scale)
    )
    design = np.asarray(_DATA["design"])
    fixed_coefficients = np.asarray(_DATA["fixed_coefficients"])
    qbase = np.asarray(_DATA["qbase"])
    contrast = np.asarray(_DATA["contrasts"][cell.target_dimension])
    true_random = true_random_covariance(
        qbase, contrast, cell.target_variance, cell.kappa
    )
    true_random_cholesky = _psd_cholesky(true_random)
    analytic_target, analytic_nuisance = projected_scales(true_random, qbase, contrast)
    normalizer = target_normalizer(qbase, contrast)
    observations = draw_population(
        rng, design, fixed_coefficients, true_random_cholesky, covariance_cholesky
    )

    test_indices = rng.integers(0, len(means), size=cell.test_subjects)
    test_design = design[test_indices]
    test_covariances = covariances[test_indices]
    test_cholesky = covariance_cholesky[test_indices]
    test_observations = draw_population(
        rng, test_design, fixed_coefficients, true_random_cholesky, test_cholesky
    )
    oracle = oracle_score(
        test_observations,
        test_design,
        test_covariances,
        fixed_coefficients,
        true_random,
        contrast,
    )

    raw = target_direction_diagnostic(
        observations, covariances, design, contrast, qbase, project_psd=False
    )
    psd = target_direction_diagnostic(
        observations, covariances, design, contrast, qbase, project_psd=True
    )
    diagnostic = {
        "target_dimension": cell.target_dimension,
        "kappa": cell.kappa,
        "target_variance": cell.target_variance,
        "observation_scale": cell.observation_scale,
        "replication": cell.replication,
        "seed": cell.seed,
        "analytic_target_variance": analytic_target,
        "analytic_nuisance_variance": analytic_nuisance,
        "analytic_kappa": analytic_nuisance / analytic_target,
        "raw_target_variance": raw.target_variance,
        "raw_nuisance_variance": raw.nuisance_variance,
        "raw_kappa": raw.kappa,
        "raw_min_eigenvalue": raw.raw_min_eigenvalue,
        "psd_target_variance": psd.target_variance,
        "psd_nuisance_variance": psd.nuisance_variance,
        "psd_kappa": psd.kappa,
        "oracle_mean_log_score": oracle,
    }

    rows: list[dict[str, object]] = []
    for workflow in WORKFLOWS:
        fit, target_space = fit_candidate(
            observations, design, covariances, qbase, contrast, workflow
        )
        if target_space:
            fitted_target = fit.random_covariance
        else:
            fitted_target = contrast @ fit.random_covariance @ contrast.T
        normalized_fitted = _sym(normalizer @ fitted_target @ normalizer.T)
        estimated_target_variance = float(np.trace(normalized_fitted)) / cell.target_dimension
        score = score_candidate(
            fit,
            target_space,
            test_observations,
            test_design,
            test_covariances,
            contrast,
        )
        covered, coverage_total = coefficient_coverage(
            fit,
            target_space,
            fixed_coefficients,
            contrast,
            normalizer,
            means.shape[1],
        )
        rows.append(
            {
                "target_dimension": cell.target_dimension,
                "kappa": cell.kappa,
                "target_variance": cell.target_variance,
                "observation_scale": cell.observation_scale,
                "replication": cell.replication,
                "seed": cell.seed,
                "workflow": workflow,
                "estimated_target_variance": estimated_target_variance,
                "target_variance_error": estimated_target_variance - cell.target_variance,
                "mean_test_log_score": score,
                "oracle_mean_log_score": oracle,
                "predictive_regret": oracle - score,
                "coverage_count": covered,
                "coverage_total": coverage_total,
                "converged": int(fit.converged),
                "boundary_fraction": fit.boundary_fraction,
                "component_count": len(fit.component_weights),
            }
        )
    return rows, diagnostic


def build_cells(args: argparse.Namespace) -> list[Cell]:
    cells = []
    seed_sequence = np.random.SeedSequence(args.seed)
    total = (
        len(args.target_dims)
        * len(args.kappas)
        * len(args.target_variances)
        * len(args.observation_scales)
        * args.reps
    )
    children = iter(seed_sequence.spawn(total))
    for target_dimension in args.target_dims:
        for kappa in args.kappas:
            for target_variance in args.target_variances:
                for observation_scale in args.observation_scales:
                    for replication in range(args.reps):
                        child = next(children)
                        cells.append(
                            Cell(
                                target_dimension=target_dimension,
                                kappa=kappa,
                                target_variance=target_variance,
                                observation_scale=observation_scale,
                                replication=replication,
                                seed=int(child.generate_state(1, dtype=np.uint32)[0]),
                                test_subjects=args.test_subjects,
                            )
                        )
    return cells


def cell_key(cell: Cell) -> tuple[int, float, float, float, int]:
    return (
        cell.target_dimension,
        float(cell.kappa),
        float(cell.target_variance),
        float(cell.observation_scale),
        cell.replication,
    )


def completed_keys(path: Path) -> set[tuple[int, float, float, float, int]]:
    if not path.exists():
        return set()
    rows = pd.read_csv(path)
    required = {
        "target_dimension",
        "kappa",
        "target_variance",
        "observation_scale",
        "replication",
    }
    if not required.issubset(rows.columns):
        raise ValueError(f"Checkpoint schema is invalid: {path}")
    return {
        (
            int(row.target_dimension),
            float(row.kappa),
            float(row.target_variance),
            float(row.observation_scale),
            int(row.replication),
        )
        for row in rows.itertuples(index=False)
    }


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def summarize_models(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(
        [
            "target_dimension",
            "kappa",
            "target_variance",
            "observation_scale",
            "workflow",
        ],
        as_index=False,
    )
    summary = grouped.agg(
        replications=("replication", "nunique"),
        mean_estimated_target_variance=("estimated_target_variance", "mean"),
        sd_estimated_target_variance=("estimated_target_variance", "std"),
        rmse_target_variance=("target_variance_error", lambda x: float(np.sqrt(np.mean(x**2)))),
        mean_test_log_score=("mean_test_log_score", "mean"),
        mean_predictive_regret=("predictive_regret", "mean"),
        convergence_rate=("converged", "mean"),
        mean_boundary_fraction=("boundary_fraction", "mean"),
        coverage_count=("coverage_count", "sum"),
        coverage_total=("coverage_total", "sum"),
    )
    summary["coverage"] = summary["coverage_count"] / summary["coverage_total"]
    return summary


def summarize_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(
            ["target_dimension", "kappa", "target_variance", "observation_scale"],
            as_index=False,
        )
        .agg(
            replications=("replication", "nunique"),
            analytic_target_variance=("analytic_target_variance", "mean"),
            analytic_nuisance_variance=("analytic_nuisance_variance", "mean"),
            analytic_kappa=("analytic_kappa", "mean"),
            mean_raw_kappa=("raw_kappa", "mean"),
            median_raw_kappa=("raw_kappa", "median"),
            mean_psd_kappa=("psd_kappa", "mean"),
            median_psd_kappa=("psd_kappa", "median"),
            raw_non_psd_fraction=("raw_min_eigenvalue", lambda x: float(np.mean(x < -1e-10))),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--covariates", type=Path, required=True)
    parser.add_argument("--target-dims", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--kappas", type=float, nargs="+", default=[0.25, 1.0, 5.0])
    parser.add_argument("--target-variances", type=float, nargs="+", default=[0.04])
    parser.add_argument("--observation-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--test-subjects", type=int, default=250)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--tag", default="finger_smoke_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if any(target_dimension not in (1, 2) for target_dimension in args.target_dims):
        raise ValueError("FingerData anchor currently supports target dimensions 1 and 2.")
    if args.reps < 1 or args.test_subjects < 1:
        raise ValueError("reps and test-subjects must be positive.")
    if any(scale <= 0 for scale in args.observation_scales):
        raise ValueError("observation scales must be positive.")

    loaded = load_anchor(args.input, args.covariates)
    worker_data = {
        key: loaded[key]
        for key in (
            "means",
            "covariances",
            "covariance_cholesky",
            "qbase",
            "design",
            "design_names",
            "fixed_coefficients",
            "contrasts",
        )
    }
    cells = build_cells(args)
    output_prefix = HERE / args.tag
    partial_models = Path(f"{output_prefix}_model_rows.partial.csv")
    partial_diagnostics = Path(f"{output_prefix}_diagnostic_rows.partial.csv")
    if not args.resume:
        partial_models.unlink(missing_ok=True)
        partial_diagnostics.unlink(missing_ok=True)
    done = completed_keys(partial_diagnostics) if args.resume else set()
    remaining = [cell for cell in cells if cell_key(cell) not in done]
    print(
        f"Empirical-anchor cells: total={len(cells)}, completed={len(done)}, "
        f"remaining={len(remaining)}",
        flush=True,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_worker,
        initargs=(worker_data,),
    ) as executor:
        for completed, (rows, diagnostic) in enumerate(
            executor.map(run_cell, remaining, chunksize=1), start=1
        ):
            append_rows(partial_models, rows)
            append_rows(partial_diagnostics, [diagnostic])
            if completed % max(1, len(remaining) // 10) == 0 or completed == len(remaining):
                print(f"Checkpointed {len(done) + completed}/{len(cells)} cells", flush=True)

    models = pd.read_csv(partial_models)
    diagnostics = pd.read_csv(partial_diagnostics)
    observed_keys = {
        (
            int(row.target_dimension),
            float(row.kappa),
            float(row.target_variance),
            float(row.observation_scale),
            int(row.replication),
        )
        for row in diagnostics.itertuples(index=False)
    }
    expected_keys = {cell_key(cell) for cell in cells}
    if observed_keys != expected_keys or len(diagnostics) != len(expected_keys):
        raise RuntimeError(
            f"Incomplete or duplicate checkpoint: expected {len(expected_keys)} cells, "
            f"observed {len(diagnostics)} rows and {len(observed_keys)} unique keys."
        )
    model_summary = summarize_models(models)
    diagnostic_summary = summarize_diagnostics(diagnostics)
    models.to_csv(f"{output_prefix}_model_rows.csv", index=False)
    diagnostics.to_csv(f"{output_prefix}_diagnostic_rows.csv", index=False)
    model_summary.to_csv(f"{output_prefix}_model_summary.csv", index=False)
    diagnostic_summary.to_csv(f"{output_prefix}_diagnostic_summary.csv", index=False)
    configuration = {
        "input": str(args.input.resolve()),
        "target_dims": args.target_dims,
        "kappas": args.kappas,
        "target_variances": args.target_variances,
        "observation_scales": args.observation_scales,
        "reps": args.reps,
        "test_subjects": args.test_subjects,
        "workers": args.workers,
        "seed": args.seed,
        "resumed": args.resume,
        "completed_cells": len(expected_keys),
        "n_subjects": int(np.asarray(loaded["means"]).shape[0]),
        "full_dimension": int(np.asarray(loaded["means"]).shape[1]),
        "design_names": loaded["design_names"],
        "fixed_effect_source": "OLS projection of frozen empirical posterior means",
        "evaluation": "independent generated test population with empirical design/covariance templates",
    }
    Path(f"{output_prefix}_config.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    print(model_summary.to_string(index=False), flush=True)
    print(diagnostic_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
