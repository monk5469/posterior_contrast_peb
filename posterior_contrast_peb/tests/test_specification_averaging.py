"""Numerical contracts for predictive specification averaging."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PREDICTION = Path(__file__).resolve().parents[1] / "simulations" / "prediction"
if str(PREDICTION) not in sys.path:
    sys.path.insert(0, str(PREDICTION))

from specification_averaging import (
    gaussian_mixture_marginal_interval,
    gaussian_mixture_moments,
    merge_predictively_equivalent,
    mixture_log_density,
    pseudo_bma_plus_weights,
    regularized_stacking_weights,
)


def test_equivalent_candidates_are_merged() -> None:
    values = np.array([[-1.0, -1.0, -2.0], [-1.5, -1.5, -1.0]])
    merged, names, classes = merge_predictively_equivalent(
        values, ("source_single", "rotated_fitted", "target_direct")
    )
    assert merged.shape == (2, 2)
    assert classes[0] == ("source_single", "rotated_fitted")
    assert names[0] == "source_single = rotated_fitted"


def test_stacking_and_pseudo_bma_weights_are_valid() -> None:
    values = np.array([[-0.2, -2.0], [-0.1, -1.8], [-0.3, -2.1]])
    for result in (
        regularized_stacking_weights(values, ("good", "poor")),
        regularized_stacking_weights(
            values, ("good", "poor"), entropy_strength=0.0
        ),
        pseudo_bma_plus_weights(values, ("good", "poor"), bootstrap_repetitions=100),
    ):
        assert np.isclose(result.weights.sum(), 1.0)
        assert np.all(result.weights >= 0)
        assert result.weights[0] > result.weights[1]
        assert 1.0 <= result.effective_count <= 2.0
        assert np.all(np.isfinite(mixture_log_density(values, result.weights)))


def test_gaussian_mixture_moments_include_between_candidate_variation() -> None:
    means = np.array([[0.0], [2.0]])
    covariances = np.array([[[1.0]], [[1.0]]])
    mean, covariance = gaussian_mixture_moments(
        means, covariances, np.array([0.5, 0.5])
    )
    assert np.allclose(mean, [1.0])
    assert np.allclose(covariance, [[2.0]])


def test_gaussian_mixture_interval_is_exact_for_identical_components() -> None:
    means = np.array([[0.0], [0.0]])
    covariances = np.array([[[1.0]], [[1.0]]])
    lower, upper = gaussian_mixture_marginal_interval(
        means, covariances, np.array([0.25, 0.75])
    )
    assert np.allclose(lower, [-1.9599639845], atol=1e-8)
    assert np.allclose(upper, [1.9599639845], atol=1e-8)


if __name__ == "__main__":
    test_equivalent_candidates_are_merged()
    test_stacking_and_pseudo_bma_weights_are_valid()
    test_gaussian_mixture_moments_include_between_candidate_variation()
    test_gaussian_mixture_interval_is_exact_for_identical_components()
    print("All specification-averaging numerical contract tests passed.")
