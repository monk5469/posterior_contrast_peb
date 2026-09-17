"""Target-predictive helpers shared by the nested resampling analysis."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parents[1] / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

from covariance_specification import (  # noqa: E402
    fixed_effect_prediction_covariance,
    multivariate_normal_logpdf,
)


CANDIDATES = ("source_single", "source_all", "target_direct", "rotated_fitted")


def _sym(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def predictive_parameters(
    fit,
    target_space: bool,
    designs: np.ndarray,
    covariances: np.ndarray,
    contrast: np.ndarray,
    source_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    predictive_covariances = []
    for row, first_level in zip(designs, covariances):
        if target_space:
            mean = row @ fit.coefficients
            fixed = fixed_effect_prediction_covariance(
                fit.coefficient_covariance,
                row,
                np.eye(contrast.shape[0]),
                contrast.shape[0],
            )
            random_target = fit.random_covariance
        else:
            mean = contrast @ (row @ fit.coefficients)
            fixed = fixed_effect_prediction_covariance(
                fit.coefficient_covariance,
                row,
                contrast,
                source_dimension,
            )
            random_target = contrast @ fit.random_covariance @ contrast.T
        means.append(np.asarray(mean).reshape(-1))
        predictive_covariances.append(
            _sym(contrast @ first_level @ contrast.T + random_target + fixed)
        )
    return np.asarray(means), np.asarray(predictive_covariances)


def pointwise_log_densities(
    values: np.ndarray,
    candidate_parameters: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    columns = []
    for means, covariances in candidate_parameters:
        columns.append(
            [
                multivariate_normal_logpdf(value, mean, covariance)
                for value, mean, covariance in zip(values, means, covariances)
            ]
        )
    return np.asarray(columns, dtype=float).T


def group_target_moments(
    fit,
    target_space: bool,
    contrast: np.ndarray,
    normalizer: np.ndarray,
    source_dimension: int,
    design_indices: tuple[int, ...] = (1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    fitted_dimension = contrast.shape[0] if target_space else source_dimension
    map_target = normalizer if target_space else normalizer @ contrast
    estimates = fit.coefficients if target_space else fit.coefficients @ contrast.T
    estimates = estimates @ normalizer.T
    mean = estimates[np.asarray(design_indices)].reshape(-1)
    rows = []
    for design_index in design_indices:
        for target_index in range(contrast.shape[0]):
            row = np.zeros(fit.coefficients.size)
            start = design_index * fitted_dimension
            row[start : start + fitted_dimension] = map_target[target_index]
            rows.append(row)
    mapping = np.asarray(rows)
    covariance = _sym(mapping @ fit.coefficient_covariance @ mapping.T)
    return mean, covariance
