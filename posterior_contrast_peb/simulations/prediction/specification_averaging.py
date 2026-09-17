"""Predictive averaging across covariance-component specifications.

Weights are predictive-combination weights, not posterior model probabilities.
The routines accept pointwise log predictive densities from an evaluation set
that was not used to fit the candidate models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp
from scipy.stats import norm


Array = np.ndarray


@dataclass(frozen=True)
class AveragingWeights:
    names: tuple[str, ...]
    weights: Array
    objective: float
    converged: bool
    effective_count: float


def _validate_log_densities(log_densities: Array, names: tuple[str, ...]) -> Array:
    values = np.asarray(log_densities, dtype=float)
    if values.ndim != 2:
        raise ValueError("log_densities must have shape observations by candidates.")
    if values.shape[1] != len(names):
        raise ValueError("Candidate names do not match log-density columns.")
    if not np.all(np.isfinite(values)):
        raise ValueError("log_densities must be finite.")
    return values


def merge_predictively_equivalent(
    log_densities: Array,
    names: tuple[str, ...],
    *,
    absolute_tolerance: float = 1e-10,
) -> tuple[Array, tuple[str, ...], tuple[tuple[str, ...], ...]]:
    """Merge numerically identical predictive candidates before weighting."""

    values = _validate_log_densities(log_densities, names)
    representatives: list[int] = []
    classes: list[list[str]] = []
    for index, name in enumerate(names):
        matched = None
        for class_index, representative in enumerate(representatives):
            if np.allclose(
                values[:, index],
                values[:, representative],
                atol=absolute_tolerance,
                rtol=0.0,
            ):
                matched = class_index
                break
        if matched is None:
            representatives.append(index)
            classes.append([name])
        else:
            classes[matched].append(name)
    merged_names = tuple(" = ".join(group) for group in classes)
    return values[:, representatives], merged_names, tuple(tuple(group) for group in classes)


def mixture_log_density(log_densities: Array, weights: Array) -> Array:
    """Pointwise log density of a finite predictive mixture."""

    values = np.asarray(log_densities, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != weights.size:
        raise ValueError("Mixture weights do not match log-density columns.")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise ValueError("Mixture weights must lie on the probability simplex.")
    log_weights = np.full_like(weights, -np.inf)
    positive = weights > 0
    log_weights[positive] = np.log(weights[positive])
    return logsumexp(values + log_weights[None, :], axis=1)


def regularized_stacking_weights(
    log_densities: Array,
    names: tuple[str, ...],
    *,
    entropy_strength: float = 1.0,
) -> AveragingWeights:
    """Estimate stacking weights with a frozen, light entropy penalty."""

    values = _validate_log_densities(log_densities, names)
    if entropy_strength < 0:
        raise ValueError("entropy_strength must be non-negative.")
    n_candidates = values.shape[1]

    def objective(weights: Array) -> float:
        weights = np.clip(np.asarray(weights, dtype=float), 1e-15, 1.0)
        score = float(mixture_log_density(values, weights / weights.sum()).sum())
        entropy = -float(np.sum(weights * np.log(weights)))
        return -(score + entropy_strength * entropy)

    initial = np.full(n_candidates, 1.0 / n_candidates)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_candidates,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    weights /= weights.sum()
    entropy = -float(np.sum(weights[weights > 0] * np.log(weights[weights > 0])))
    return AveragingWeights(
        names=names,
        weights=weights,
        objective=-float(result.fun),
        converged=bool(result.success),
        effective_count=float(np.exp(entropy)),
    )


def pseudo_bma_plus_weights(
    log_densities: Array,
    names: tuple[str, ...],
    *,
    bootstrap_repetitions: int = 500,
    seed: int = 20260905,
) -> AveragingWeights:
    """Bayesian-bootstrap pseudo-BMA+ weights for sensitivity analysis."""

    values = _validate_log_densities(log_densities, names)
    if bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive.")
    rng = np.random.default_rng(seed)
    weights = np.zeros(values.shape[1])
    for _ in range(bootstrap_repetitions):
        observation_weights = rng.dirichlet(np.ones(values.shape[0])) * values.shape[0]
        elpd = observation_weights @ values
        candidate_weights = np.exp(elpd - np.max(elpd))
        weights += candidate_weights / candidate_weights.sum()
    weights /= bootstrap_repetitions
    entropy = -float(np.sum(weights[weights > 0] * np.log(weights[weights > 0])))
    score = float(mixture_log_density(values, weights).sum())
    return AveragingWeights(
        names=names,
        weights=weights,
        objective=score,
        converged=True,
        effective_count=float(np.exp(entropy)),
    )


def gaussian_mixture_moments(
    means: Array,
    covariances: Array,
    weights: Array,
) -> tuple[Array, Array]:
    """Return exact first two moments of a finite Gaussian mixture."""

    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if means.ndim != 2:
        raise ValueError("means must have shape candidates by dimension.")
    if covariances.shape != (means.shape[0], means.shape[1], means.shape[1]):
        raise ValueError("covariances do not match candidate means.")
    if weights.size != means.shape[0] or np.any(weights < 0):
        raise ValueError("weights do not match candidate moments.")
    weights = weights / weights.sum()
    mixture_mean = weights @ means
    mixture_covariance = np.zeros((means.shape[1], means.shape[1]))
    for weight, mean, covariance in zip(weights, means, covariances):
        difference = mean - mixture_mean
        mixture_covariance += weight * (covariance + np.outer(difference, difference))
    return mixture_mean, 0.5 * (mixture_covariance + mixture_covariance.T)


def gaussian_mixture_marginal_interval(
    means: Array,
    covariances: Array,
    weights: Array,
    *,
    probability: float = 0.95,
) -> tuple[Array, Array]:
    """Exact equal-tail marginal intervals for a finite Gaussian mixture."""

    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if means.ndim != 2 or covariances.shape != (
        means.shape[0],
        means.shape[1],
        means.shape[1],
    ):
        raise ValueError("Candidate moments have incompatible shapes.")
    if weights.size != means.shape[0] or np.any(weights < 0):
        raise ValueError("weights do not match candidate moments.")
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie between zero and one.")
    weights = weights / weights.sum()
    standard_deviations = np.sqrt(
        np.maximum(np.diagonal(covariances, axis1=1, axis2=2), 1e-15)
    )
    tail = 0.5 * (1.0 - probability)
    lower = np.empty(means.shape[1])
    upper = np.empty(means.shape[1])
    for dimension in range(means.shape[1]):
        location = means[:, dimension]
        scale = standard_deviations[:, dimension]
        left = float(np.min(location - 12.0 * scale))
        right = float(np.max(location + 12.0 * scale))

        def cdf(value: float) -> float:
            return float(np.sum(weights * norm.cdf((value - location) / scale)))

        lower[dimension] = brentq(lambda value: cdf(value) - tail, left, right)
        upper[dimension] = brentq(
            lambda value: cdf(value) - (1.0 - tail), left, right
        )
    return lower, upper
