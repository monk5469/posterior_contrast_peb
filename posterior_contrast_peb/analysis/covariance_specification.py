"""Target-focused covariance-specification diagnostics and prediction.

The routines in this module operate on subject-level Gaussian posterior
summaries.  They deliberately separate two questions:

1. how random variation is allocated between a prespecified target subspace
   and its orthogonal complement; and
2. how well a fitted covariance-component specification predicts the same
   held-out target quantity.

LOSO scores are empirical-Bayes plug-in predictive scores.  Hyperparameter
uncertainty is not silently treated as integrated uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize


Array = np.ndarray


def _sym(matrix: Array) -> Array:
    matrix = np.asarray(matrix, dtype=float)
    return 0.5 * (matrix + matrix.T)


def _inverse_sqrt(matrix: Array, relative_tolerance: float) -> Array:
    values, vectors = np.linalg.eigh(_sym(matrix))
    cutoff = relative_tolerance * max(float(values.max(initial=0.0)), 1.0)
    if np.any(values < -cutoff):
        raise ValueError("Matrix is not positive semidefinite within tolerance.")
    keep = values > cutoff
    if not np.any(keep):
        raise ValueError("Matrix has zero numerical rank.")
    return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T


@dataclass(frozen=True)
class TargetGeometry:
    """Whitening and orthonormal target/complement bases on Qbase support."""

    support_basis: Array
    support_values: Array
    whitening: Array
    colouring: Array
    target_basis: Array
    nuisance_basis: Array
    full_dimension: int
    support_dimension: int
    target_dimension: int
    eigenvalue_cutoff: float


def target_geometry(
    qbase: Array,
    contrast: Array,
    relative_tolerance: float = 1e-10,
) -> TargetGeometry:
    """Construct a scale-aware target basis on the support of ``qbase``.

    The support coordinates satisfy ``theta = colouring @ z``.  The rows of
    ``target_basis`` span the target-sensitive directions in these whitened
    coordinates and are orthonormal.  A target with rank outside the support
    is rejected rather than silently reduced.
    """

    qbase = _sym(np.asarray(qbase, dtype=float))
    contrast = np.atleast_2d(np.asarray(contrast, dtype=float))
    if qbase.ndim != 2 or qbase.shape[0] != qbase.shape[1]:
        raise ValueError("qbase must be square.")
    if contrast.shape[1] != qbase.shape[0]:
        raise ValueError("contrast and qbase dimensions do not match.")
    if np.linalg.matrix_rank(contrast) != contrast.shape[0]:
        raise ValueError("contrast rows must be linearly independent.")

    values, vectors = np.linalg.eigh(qbase)
    cutoff = relative_tolerance * max(float(values.max(initial=0.0)), 1.0)
    if np.any(values < -cutoff):
        raise ValueError("qbase is not positive semidefinite within tolerance.")
    keep = values > cutoff
    support_values = values[keep]
    support_basis = vectors[:, keep]
    support_dimension = int(keep.sum())
    if support_dimension == 0:
        raise ValueError("qbase has empty numerical support.")

    colouring = support_basis * np.sqrt(support_values)
    whitening = (support_basis / np.sqrt(support_values)).T
    sensitivity = contrast @ colouring
    target_dimension = contrast.shape[0]
    if np.linalg.matrix_rank(sensitivity, tol=cutoff) != target_dimension:
        raise ValueError("At least one target direction lies outside qbase support.")

    gram = sensitivity @ sensitivity.T
    target_basis = _inverse_sqrt(gram, relative_tolerance) @ sensitivity
    nuisance_basis = null_space(target_basis, rcond=relative_tolerance).T
    if target_basis.shape[0] + nuisance_basis.shape[0] != support_dimension:
        raise RuntimeError("Target and nuisance bases do not span qbase support.")

    return TargetGeometry(
        support_basis=support_basis,
        support_values=support_values,
        whitening=whitening,
        colouring=colouring,
        target_basis=target_basis,
        nuisance_basis=nuisance_basis,
        full_dimension=qbase.shape[0],
        support_dimension=support_dimension,
        target_dimension=target_dimension,
        eigenvalue_cutoff=cutoff,
    )


def _residual_projector(design: Array) -> tuple[Array, int]:
    design = np.atleast_2d(np.asarray(design, dtype=float))
    rank = int(np.linalg.matrix_rank(design))
    if rank == 0 or rank >= design.shape[0]:
        raise ValueError("design must have positive rank smaller than N.")
    projector = np.eye(design.shape[0]) - design @ np.linalg.pinv(design)
    return _sym(projector), design.shape[0] - rank


@dataclass(frozen=True)
class DirectionDiagnostic:
    target_variance: float
    nuisance_variance: float
    kappa: float
    predicted_shared_ratio: float
    target_stable: bool
    raw_min_eigenvalue: float
    used_psd_projection: bool
    support_dimension: int
    target_dimension: int
    corrected_covariance: Array


def target_direction_diagnostic(
    posterior_means: Array,
    posterior_covariances: Array,
    design: Array,
    contrast: Array,
    qbase: Array,
    *,
    relative_tolerance: float = 1e-10,
    target_variance_tolerance: float = 1e-10,
    project_psd: bool = False,
) -> DirectionDiagnostic:
    """Estimate target/nuisance variance allocation after error subtraction.

    Under independent subjects, a correctly specified fixed-effect design,
    and a common between-subject covariance, the subtraction

        sum_s M_X[s,s] Omega_s / trace(M_X)

    is the exact expectation of the first-level error contribution to the
    residual scatter, including heteroscedastic first-level covariances.
    """

    means = np.asarray(posterior_means, dtype=float)
    covariances = np.asarray(posterior_covariances, dtype=float)
    if means.ndim != 2:
        raise ValueError("posterior_means must be N by D.")
    n_subjects, dimension = means.shape
    if covariances.shape != (n_subjects, dimension, dimension):
        raise ValueError("posterior_covariances must have shape N by D by D.")
    if np.asarray(design).shape[0] != n_subjects:
        raise ValueError("design must have N rows.")

    geometry = target_geometry(qbase, contrast, relative_tolerance)
    design = np.asarray(design, dtype=float)
    design_rank = int(np.linalg.matrix_rank(design))
    if design_rank == 0 or design_rank >= n_subjects:
        raise ValueError("design must have positive rank smaller than N.")
    design_pseudoinverse = np.linalg.pinv(design)
    residual_df = n_subjects - design_rank
    whitened_means = means @ geometry.whitening.T
    whitened_covariances = np.stack(
        [geometry.whitening @ _sym(covariance) @ geometry.whitening.T for covariance in covariances]
    )

    residuals = whitened_means - design @ (design_pseudoinverse @ whitened_means)
    residual_scatter = residuals.T @ residuals / residual_df
    residual_projector_diagonal = 1.0 - np.einsum(
        "ij,ji->i", design, design_pseudoinverse
    )
    error_contribution = np.einsum(
        "s,sij->ij", residual_projector_diagonal, whitened_covariances
    ) / residual_df
    corrected = _sym(residual_scatter - error_contribution)
    eigenvalues, eigenvectors = np.linalg.eigh(corrected)
    raw_minimum = float(eigenvalues.min())
    if project_psd:
        corrected = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
        corrected = _sym(corrected)

    target_basis = geometry.target_basis
    nuisance_basis = geometry.nuisance_basis
    target_variance = float(np.trace(target_basis @ corrected @ target_basis.T)) / geometry.target_dimension
    nuisance_dimension = nuisance_basis.shape[0]
    nuisance_variance = (
        float(np.trace(nuisance_basis @ corrected @ nuisance_basis.T)) / nuisance_dimension
        if nuisance_dimension
        else float("nan")
    )
    stable = bool(np.isfinite(target_variance) and target_variance > target_variance_tolerance)
    if stable and nuisance_dimension:
        kappa = nuisance_variance / target_variance
        predicted = (
            geometry.target_dimension + nuisance_dimension * kappa
        ) / geometry.support_dimension
    elif stable:
        kappa = float("nan")
        predicted = 1.0
    else:
        kappa = float("nan")
        predicted = float("nan")

    return DirectionDiagnostic(
        target_variance=target_variance,
        nuisance_variance=nuisance_variance,
        kappa=float(kappa),
        predicted_shared_ratio=float(predicted),
        target_stable=stable,
        raw_min_eigenvalue=raw_minimum,
        used_psd_projection=project_psd,
        support_dimension=geometry.support_dimension,
        target_dimension=geometry.target_dimension,
        corrected_covariance=corrected,
    )


def bootstrap_direction_diagnostic(
    posterior_means: Array,
    posterior_covariances: Array,
    design: Array,
    contrast: Array,
    qbase: Array,
    *,
    repetitions: int = 500,
    seed: int = 20260904,
    project_psd: bool = False,
    strata: Array | None = None,
) -> dict[str, float]:
    """Subject bootstrap for ``kappa`` and the shared-component ratio.

    When ``strata`` is supplied, subjects are resampled independently within
    each stratum. This preserves the group counts and avoids turning a group
    contrast into a changing bootstrap design.
    """

    means = np.asarray(posterior_means)
    covariances = np.asarray(posterior_covariances)
    design = np.asarray(design)
    if strata is None:
        stratum_indices = [np.arange(means.shape[0])]
    else:
        strata = np.asarray(strata)
        if strata.ndim != 1 or strata.shape[0] != means.shape[0]:
            raise ValueError("strata must be a length-N vector.")
        stratum_indices = [
            np.flatnonzero(strata == value) for value in np.unique(strata)
        ]
        if any(len(indices) == 0 for indices in stratum_indices):
            raise ValueError("Every bootstrap stratum must contain a subject.")
    rng = np.random.default_rng(seed)
    kappas: list[float] = []
    ratios: list[float] = []
    unstable = 0
    nonpositive_target = 0
    nonpositive_nuisance = 0
    rank_failures = 0
    for _ in range(repetitions):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in stratum_indices]
        )
        try:
            diagnostic = target_direction_diagnostic(
                means[indices],
                covariances[indices],
                design[indices],
                contrast,
                qbase,
                project_psd=project_psd,
            )
        except ValueError:
            rank_failures += 1
            continue
        if not diagnostic.target_stable:
            nonpositive_target += 1
        elif not np.isfinite(diagnostic.nuisance_variance) or diagnostic.nuisance_variance <= 0.0:
            nonpositive_nuisance += 1
        if (
            diagnostic.target_stable
            and np.isfinite(diagnostic.nuisance_variance)
            and diagnostic.nuisance_variance > 0.0
            and np.isfinite(diagnostic.kappa)
            and diagnostic.kappa > 0.0
        ):
            kappas.append(diagnostic.kappa)
            ratios.append(diagnostic.predicted_shared_ratio)
        else:
            unstable += 1

    if not kappas:
        return {
            "valid_repetitions": 0.0,
            "unstable_fraction": 1.0,
            "rank_failure_fraction": rank_failures / repetitions,
            "nonpositive_target_fraction": nonpositive_target / repetitions,
            "nonpositive_nuisance_fraction": nonpositive_nuisance / repetitions,
            "kappa_median": float("nan"),
            "kappa_lower": float("nan"),
            "kappa_upper": float("nan"),
            "ratio_median": float("nan"),
            "ratio_lower": float("nan"),
            "ratio_upper": float("nan"),
        }
    kappa_array = np.asarray(kappas)
    ratio_array = np.asarray(ratios)
    return {
        "valid_repetitions": float(len(kappas)),
        "unstable_fraction": unstable / repetitions,
        "rank_failure_fraction": rank_failures / repetitions,
        "nonpositive_target_fraction": nonpositive_target / repetitions,
        "nonpositive_nuisance_fraction": nonpositive_nuisance / repetitions,
        "kappa_median": float(np.median(kappa_array)),
        "kappa_lower": float(np.quantile(kappa_array, 0.025)),
        "kappa_upper": float(np.quantile(kappa_array, 0.975)),
        "ratio_median": float(np.median(ratio_array)),
        "ratio_lower": float(np.quantile(ratio_array, 0.025)),
        "ratio_upper": float(np.quantile(ratio_array, 0.975)),
    }


def classify_kappa_interval(
    lower: float,
    upper: float,
    *,
    equivalence_factor: float = 1.25,
    valid_fraction: float = 1.0,
    minimum_valid_fraction: float = 0.8,
) -> int:
    """Classify a bootstrap interval using a frozen practical-equivalence band.

    Returns ``-1`` for target-dominant allocation, ``0`` for an interval fully
    contained in the equivalence band, ``1`` for complement-dominant
    allocation, ``2`` for an inconclusive interval, and ``9`` for an unstable
    diagnostic.
    """

    if equivalence_factor <= 1.0:
        raise ValueError("equivalence_factor must exceed one.")
    if (
        valid_fraction < minimum_valid_fraction
        or not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower > upper
    ):
        return 9
    if lower <= 0.0:
        return 2
    lower_bound = 1.0 / equivalence_factor
    upper_bound = equivalence_factor
    if upper < lower_bound:
        return -1
    if lower > upper_bound:
        return 1
    if lower >= lower_bound and upper <= upper_bound:
        return 0
    return 2


def contrast_pair_matrix(full_dimension: int, target_dimension: int) -> Array:
    """Orthonormal pairwise-difference targets used by the validation grid."""

    if full_dimension < 2 * target_dimension:
        raise ValueError("full_dimension must be at least twice target_dimension.")
    contrast = np.zeros((target_dimension, full_dimension))
    for row in range(target_dimension):
        contrast[row, 2 * row] = 1.0 / math.sqrt(2.0)
        contrast[row, 2 * row + 1] = -1.0 / math.sqrt(2.0)
    return contrast


def covariance_components(full_dimension: int, target: Array, family: str) -> list[Array]:
    """Return candidate covariance components for the simulation study."""

    identity = np.eye(full_dimension)
    if family == "source_single":
        return [identity]
    if family == "source_all":
        return [np.diag(np.eye(full_dimension)[index]) for index in range(full_dimension)]
    projector = target.T @ np.linalg.inv(target @ target.T) @ target
    if family == "rotated_fitted":
        return [_sym(projector), _sym(identity - projector)]
    if family == "target_direct":
        return [np.eye(target.shape[0])]
    raise ValueError(f"Unknown component family: {family}")


@dataclass(frozen=True)
class RemlFit:
    component_weights: Array
    random_covariance: Array
    total_covariance: Array
    coefficients: Array
    coefficient_row_covariance: Array
    converged: bool
    boundary_fraction: float
    objective: float


def _reml_objective(
    weights: Array,
    scatter: Array,
    residual_df: int,
    observation_covariance: Array,
    components: Sequence[Array],
) -> tuple[float, Array]:
    total = observation_covariance.copy()
    for weight, component in zip(weights, components):
        total += weight * component
    total = _sym(total)
    sign, logdet = np.linalg.slogdet(total)
    if sign <= 0:
        return 1e100, np.full_like(weights, 1e50)
    inverse = np.linalg.inv(total)
    value = 0.5 * (residual_df * logdet + float(np.trace(inverse @ scatter)))
    middle = inverse @ scatter @ inverse
    gradient = np.asarray(
        [
            0.5
            * (
                residual_df * float(np.trace(inverse @ component))
                - float(np.trace(middle @ component))
            )
            for component in components
        ]
    )
    return float(value), gradient


def _orthogonal_projector_weights(
    scatter: Array,
    residual_df: int,
    observation_covariance: Array,
    components: Sequence[Array],
    tolerance: float = 1e-9,
) -> Array | None:
    """Exact constrained REML solution for isotropic noise and projectors."""

    dimension = observation_covariance.shape[0]
    noise_scale = float(np.trace(observation_covariance) / dimension)
    if not np.allclose(
        observation_covariance, noise_scale * np.eye(dimension), atol=tolerance, rtol=tolerance
    ):
        return None
    projectors = [_sym(np.asarray(component, dtype=float)) for component in components]
    for index, projector in enumerate(projectors):
        if not np.allclose(projector @ projector, projector, atol=tolerance, rtol=tolerance):
            return None
        for previous in projectors[:index]:
            if not np.allclose(projector @ previous, 0.0, atol=tolerance, rtol=tolerance):
                return None
    weights = []
    for projector in projectors:
        rank = float(np.trace(projector))
        if rank <= tolerance:
            return None
        empirical_scale = float(np.trace(projector @ scatter)) / (residual_df * rank)
        weights.append(max(empirical_scale - noise_scale, 0.0))
    return np.asarray(weights)


def fit_common_covariance_reml(
    observations: Array,
    design: Array,
    observation_covariance: Array,
    components: Sequence[Array],
) -> RemlFit:
    """Fit non-negative covariance components for a common within-subject V."""

    observations = np.asarray(observations, dtype=float)
    design = np.asarray(design, dtype=float)
    observation_covariance = _sym(np.asarray(observation_covariance, dtype=float))
    if observations.ndim != 2 or design.shape[0] != observations.shape[0]:
        raise ValueError("observations and design must share N rows.")
    dimension = observations.shape[1]
    if observation_covariance.shape != (dimension, dimension):
        raise ValueError("observation_covariance has the wrong shape.")
    if any(np.asarray(component).shape != (dimension, dimension) for component in components):
        raise ValueError("Every covariance component must match the data dimension.")

    projector, residual_df = _residual_projector(design)
    scatter = observations.T @ projector @ observations
    weights = _orthogonal_projector_weights(
        scatter, residual_df, observation_covariance, components
    )
    if weights is None:
        empirical = np.diag(scatter) / residual_df - np.diag(observation_covariance)
        initial = max(float(np.median(np.maximum(empirical, 0.0))), 1e-6)
        upper = max(2.0, 20.0 * float(np.max(np.diag(scatter) / residual_df)))
        result = minimize(
            lambda candidate: _reml_objective(
                candidate, scatter, residual_df, observation_covariance, components
            ),
            x0=np.full(len(components), initial),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, upper)] * len(components),
            options={"maxiter": 240, "ftol": 1e-11, "gtol": 1e-8, "maxls": 50},
        )
        weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
        converged = bool(result.success)
        objective = float(result.fun)
    else:
        objective = float("nan")
        converged = True
    random_covariance = np.zeros_like(observation_covariance)
    for weight, component in zip(weights, components):
        random_covariance += weight * component
    total_covariance = _sym(observation_covariance + random_covariance)
    xtx_inverse = np.linalg.inv(design.T @ design)
    coefficients = xtx_inverse @ design.T @ observations
    return RemlFit(
        component_weights=weights,
        random_covariance=_sym(random_covariance),
        total_covariance=total_covariance,
        coefficients=coefficients,
        coefficient_row_covariance=xtx_inverse,
        converged=converged,
        boundary_fraction=float(np.mean(weights <= 1e-8)),
        objective=objective,
    )


def multivariate_normal_logpdf(value: Array, mean: Array, covariance: Array) -> float:
    value = np.asarray(value, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    covariance = _sym(np.asarray(covariance, dtype=float))
    difference = value - mean
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("Predictive covariance is not positive definite.")
    quadratic = float(difference @ np.linalg.solve(covariance, difference))
    return -0.5 * (len(value) * math.log(2.0 * math.pi) + logdet + quadratic)


@dataclass(frozen=True)
class LosoResult:
    workflows: tuple[str, ...]
    pointwise_log_density: Array
    elpd: Array
    pairwise_delta: Array
    pairwise_se: Array
    compatible: Array
    best_index: int


@dataclass(frozen=True)
class HeteroscedasticRemlFit:
    component_weights: Array
    random_covariance: Array
    coefficients: Array
    coefficient_covariance: Array
    converged: bool
    boundary_fraction: float
    objective: float


@dataclass(frozen=True)
class TwoScaleDiagnostic:
    target_variance: float
    nuisance_variance: float
    kappa: float
    kappa_lower: float
    kappa_upper: float
    converged: bool
    boundary: bool
    information_stable: bool
    objective: float


@dataclass(frozen=True)
class ProfileBoundaryDiagnostic:
    target_variance: float
    nuisance_variance: float
    kappa: float
    lower_boundary_statistic: float
    upper_boundary_statistic: float
    critical_value: float
    decision: int
    converged: bool
    target_boundary: bool
    nuisance_boundary: bool
    objective: float


def _heteroscedastic_reml_value_gradient(
    weights: Array,
    observations: Array,
    design: Array,
    observation_covariances: Array,
    components: Sequence[Array],
    *,
    details: bool = False,
):
    """Restricted Gaussian likelihood with subject-specific covariance."""

    n_subjects, dimension = observations.shape
    predictors = design.shape[1]
    random_covariance = np.zeros((dimension, dimension))
    for weight, component in zip(weights, components):
        random_covariance += weight * component
    random_covariance = _sym(random_covariance)

    information = np.zeros((predictors * dimension, predictors * dimension))
    rhs = np.zeros((predictors, dimension))
    quadratic = 0.0
    logdet_total = 0.0
    inverses: list[Array] = []
    for subject in range(n_subjects):
        total = _sym(observation_covariances[subject] + random_covariance)
        sign, logdet = np.linalg.slogdet(total)
        if sign <= 0:
            bad_gradient = np.full(len(components), 1e50)
            return (1e100, bad_gradient, None) if details else (1e100, bad_gradient)
        inverse = np.linalg.inv(total)
        inverses.append(inverse)
        x = design[subject]
        information += np.kron(np.outer(x, x), inverse)
        weighted = inverse @ observations[subject]
        rhs += x[:, None] * weighted[None, :]
        quadratic += float(observations[subject] @ weighted)
        logdet_total += float(logdet)

    sign_information, logdet_information = np.linalg.slogdet(information)
    if sign_information <= 0:
        bad_gradient = np.full(len(components), 1e50)
        return (1e100, bad_gradient, None) if details else (1e100, bad_gradient)
    coefficient_covariance = np.linalg.inv(information)
    coefficient_vector = coefficient_covariance @ rhs.reshape(-1)
    coefficients = coefficient_vector.reshape(predictors, dimension)
    residuals = observations - design @ coefficients
    restricted_quadratic = quadratic - float(rhs.reshape(-1) @ coefficient_vector)
    value = 0.5 * (logdet_total + logdet_information + restricted_quadratic)

    gradient = np.zeros(len(components))
    for subject in range(n_subjects):
        x = design[subject]
        design_block = np.kron(x.reshape(1, -1), np.eye(dimension))
        fixed_prediction_covariance = design_block @ coefficient_covariance @ design_block.T
        inverse = inverses[subject]
        weighted_residual = inverse @ residuals[subject]
        for component_index, component in enumerate(components):
            transformed = inverse @ component @ inverse
            gradient[component_index] += 0.5 * (
                float(np.trace(inverse @ component))
                - float(np.trace(fixed_prediction_covariance @ transformed))
                - float(weighted_residual @ component @ weighted_residual)
            )

    if not details:
        return float(value), gradient
    payload = {
        "random_covariance": random_covariance,
        "coefficients": coefficients,
        "coefficient_covariance": coefficient_covariance,
    }
    return float(value), gradient, payload


def conditional_heteroscedastic_group_posterior(
    weights: Array,
    observations: Array,
    design: Array,
    observation_covariances: Array,
    components: Sequence[Array],
) -> tuple[float, Array, Array, Array]:
    """Evaluate the conditional group posterior at fixed component weights.

    This public wrapper is used when integrating group-effect uncertainty over
    an approximate covariance-component posterior. It does not re-estimate the
    weights and therefore keeps conditional and hyperparameter uncertainty
    separate.
    """

    normalized_components = [
        _sym(np.asarray(component, dtype=float)) for component in components
    ]
    value, _, payload = _heteroscedastic_reml_value_gradient(
        np.asarray(weights, dtype=float),
        np.asarray(observations, dtype=float),
        np.asarray(design, dtype=float),
        np.asarray(observation_covariances, dtype=float),
        normalized_components,
        details=True,
    )
    if payload is None:
        raise ValueError("The supplied covariance-component weights are invalid.")
    return (
        float(value),
        payload["coefficients"],
        payload["coefficient_covariance"],
        payload["random_covariance"],
    )


def fit_heteroscedastic_reml(
    observations: Array,
    design: Array,
    observation_covariances: Array,
    components: Sequence[Array],
) -> HeteroscedasticRemlFit:
    """Fit covariance components with full subject-specific first-level V."""

    observations = np.asarray(observations, dtype=float)
    design = np.asarray(design, dtype=float)
    observation_covariances = np.asarray(observation_covariances, dtype=float)
    if observations.ndim != 2:
        raise ValueError("observations must be N by D.")
    n_subjects, dimension = observations.shape
    if design.shape[0] != n_subjects:
        raise ValueError("design and observations must share N rows.")
    if observation_covariances.shape != (n_subjects, dimension, dimension):
        raise ValueError("observation_covariances must have shape N by D by D.")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("design columns must be linearly independent.")
    normalized_components = [_sym(np.asarray(component, dtype=float)) for component in components]
    if any(component.shape != (dimension, dimension) for component in normalized_components):
        raise ValueError("Every covariance component must match D.")

    projector, residual_df = _residual_projector(design)
    scatter = observations.T @ projector @ observations / residual_df
    mean_observation = np.mean(observation_covariances, axis=0)
    excess_scale = max(
        float(np.trace(scatter - mean_observation)) / dimension,
        1e-5,
    )
    component_scales = np.asarray(
        [max(float(np.trace(component)) / dimension, 1e-12) for component in normalized_components]
    )
    initial = np.maximum(excess_scale / (len(normalized_components) * component_scales), 1e-5)
    upper = max(20.0, 100.0 * excess_scale / float(component_scales.min()))
    result = minimize(
        lambda candidate: _heteroscedastic_reml_value_gradient(
            candidate,
            observations,
            design,
            observation_covariances,
            normalized_components,
        ),
        x0=initial,
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, upper)] * len(normalized_components),
        options={"maxiter": 240, "ftol": 1e-10, "gtol": 1e-7, "maxls": 40},
    )
    if not result.success:
        retry = minimize(
            lambda candidate: _heteroscedastic_reml_value_gradient(
                candidate,
                observations,
                design,
                observation_covariances,
                normalized_components,
            ),
            x0=np.maximum(np.asarray(result.x, dtype=float), 0.0),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, upper)] * len(normalized_components),
            options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-6, "maxls": 100},
        )
        if retry.success or float(retry.fun) < float(result.fun):
            result = retry
    weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    value, _, payload = _heteroscedastic_reml_value_gradient(
        weights,
        observations,
        design,
        observation_covariances,
        normalized_components,
        details=True,
    )
    if payload is None:
        raise RuntimeError("Final heteroscedastic REML fit is invalid.")
    return HeteroscedasticRemlFit(
        component_weights=weights,
        random_covariance=payload["random_covariance"],
        coefficients=payload["coefficients"],
        coefficient_covariance=payload["coefficient_covariance"],
        converged=bool(result.success),
        boundary_fraction=float(np.mean(weights <= 1e-8)),
        objective=value,
    )


def target_complement_components(
    qbase: Array,
    contrast: Array,
    relative_tolerance: float = 1e-10,
) -> tuple[Array, Array]:
    """Return scale-aware target and orthogonal-complement components."""

    geometry = target_geometry(qbase, contrast, relative_tolerance)
    target_projector = geometry.target_basis.T @ geometry.target_basis
    nuisance_projector = geometry.nuisance_basis.T @ geometry.nuisance_basis
    target = _sym(geometry.colouring @ target_projector @ geometry.colouring.T)
    nuisance = _sym(geometry.colouring @ nuisance_projector @ geometry.colouring.T)
    return target, nuisance


def two_scale_reml_direction_diagnostic(
    posterior_means: Array,
    posterior_covariances: Array,
    design: Array,
    contrast: Array,
    qbase: Array,
    *,
    confidence_level: float = 0.95,
    log_step: float = 1e-3,
    boundary_tolerance: float = 1e-8,
) -> TwoScaleDiagnostic:
    """Estimate target/complement scales and a local likelihood interval.

    The interval is a Wald approximation on ``log(tau_N^2/tau_T^2)`` using
    the observed curvature of the restricted likelihood. Boundary solutions
    and non-positive curvature are flagged rather than silently interpreted.
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between zero and one.")
    if log_step <= 0.0:
        raise ValueError("log_step must be positive.")
    components = target_complement_components(qbase, contrast)
    fit = fit_heteroscedastic_reml(
        posterior_means, design, posterior_covariances, components
    )
    weights = np.asarray(fit.component_weights, dtype=float)
    boundary = bool(np.any(weights <= boundary_tolerance))
    if boundary or not fit.converged:
        return TwoScaleDiagnostic(
            target_variance=float(weights[0]),
            nuisance_variance=float(weights[1]),
            kappa=float(weights[1] / weights[0]) if weights[0] > 0.0 else float("nan"),
            kappa_lower=float("nan"),
            kappa_upper=float("nan"),
            converged=fit.converged,
            boundary=boundary,
            information_stable=False,
            objective=fit.objective,
        )

    eta = np.log(weights)

    def objective(log_weights: Array) -> float:
        value, _ = _heteroscedastic_reml_value_gradient(
            np.exp(log_weights),
            np.asarray(posterior_means, dtype=float),
            np.asarray(design, dtype=float),
            np.asarray(posterior_covariances, dtype=float),
            components,
        )
        return float(value)

    center = objective(eta)
    hessian = np.zeros((2, 2))
    for index in range(2):
        offset = np.zeros(2)
        offset[index] = log_step
        hessian[index, index] = (
            objective(eta + offset) - 2.0 * center + objective(eta - offset)
        ) / (log_step**2)
    first = np.array([log_step, 0.0])
    second = np.array([0.0, log_step])
    hessian[0, 1] = hessian[1, 0] = (
        objective(eta + first + second)
        - objective(eta + first - second)
        - objective(eta - first + second)
        + objective(eta - first - second)
    ) / (4.0 * log_step**2)
    eigenvalues = np.linalg.eigvalsh(_sym(hessian))
    stable = bool(np.all(np.isfinite(eigenvalues)) and np.min(eigenvalues) > 1e-8)
    kappa = float(weights[1] / weights[0])
    if stable:
        covariance = np.linalg.inv(_sym(hessian))
        contrast_log = np.array([-1.0, 1.0])
        variance_log_kappa = float(contrast_log @ covariance @ contrast_log)
        stable = bool(np.isfinite(variance_log_kappa) and variance_log_kappa > 0.0)
    if stable:
        from scipy.stats import norm

        critical = float(norm.ppf(0.5 + confidence_level / 2.0))
        half_width = critical * math.sqrt(variance_log_kappa)
        lower = float(math.exp(math.log(kappa) - half_width))
        upper = float(math.exp(math.log(kappa) + half_width))
    else:
        lower = float("nan")
        upper = float("nan")
    return TwoScaleDiagnostic(
        target_variance=float(weights[0]),
        nuisance_variance=float(weights[1]),
        kappa=kappa,
        kappa_lower=lower,
        kappa_upper=upper,
        converged=fit.converged,
        boundary=boundary,
        information_stable=stable,
        objective=fit.objective,
    )


def profile_boundary_reml_direction_diagnostic(
    posterior_means: Array,
    posterior_covariances: Array,
    design: Array,
    contrast: Array,
    qbase: Array,
    *,
    equivalence_factor: float = 1.25,
    confidence_level: float = 0.95,
    boundary_tolerance: float = 1e-8,
) -> ProfileBoundaryDiagnostic:
    """Classify target/complement allocation by profile-likelihood boundaries.

    The unrestricted two-scale fit is compared with one-scale fits constrained
    to each edge of the frozen practical-equivalence band. The returned class
    uses the same codes as :func:`classify_kappa_interval`.
    """

    from scipy.stats import chi2

    if equivalence_factor <= 1.0:
        raise ValueError("equivalence_factor must exceed one.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between zero and one.")
    target_component, nuisance_component = target_complement_components(
        qbase, contrast
    )
    full = fit_heteroscedastic_reml(
        posterior_means,
        design,
        posterior_covariances,
        (target_component, nuisance_component),
    )
    weights = np.asarray(full.component_weights, dtype=float)
    target_boundary = bool(weights[0] <= boundary_tolerance)
    nuisance_boundary = bool(weights[1] <= boundary_tolerance)
    if target_boundary and nuisance_boundary:
        kappa = float("nan")
    elif target_boundary:
        kappa = float("inf")
    elif nuisance_boundary:
        kappa = 0.0
    else:
        kappa = float(weights[1] / weights[0])

    lower_bound = 1.0 / equivalence_factor
    upper_bound = equivalence_factor
    lower_fit = fit_heteroscedastic_reml(
        posterior_means,
        design,
        posterior_covariances,
        (target_component + lower_bound * nuisance_component,),
    )
    upper_fit = fit_heteroscedastic_reml(
        posterior_means,
        design,
        posterior_covariances,
        (target_component + upper_bound * nuisance_component,),
    )
    lower_statistic = max(0.0, 2.0 * (lower_fit.objective - full.objective))
    upper_statistic = max(0.0, 2.0 * (upper_fit.objective - full.objective))
    critical = float(chi2.ppf(confidence_level, df=1))
    converged = bool(full.converged and lower_fit.converged and upper_fit.converged)
    if not converged or np.isnan(kappa):
        decision = 9
    elif kappa < lower_bound and lower_statistic > critical:
        decision = -1
    elif kappa > upper_bound and upper_statistic > critical:
        decision = 1
    elif (
        lower_bound <= kappa <= upper_bound
        and lower_statistic > critical
        and upper_statistic > critical
    ):
        decision = 0
    else:
        decision = 2
    return ProfileBoundaryDiagnostic(
        target_variance=float(weights[0]),
        nuisance_variance=float(weights[1]),
        kappa=kappa,
        lower_boundary_statistic=float(lower_statistic),
        upper_boundary_statistic=float(upper_statistic),
        critical_value=critical,
        decision=decision,
        converged=converged,
        target_boundary=target_boundary,
        nuisance_boundary=nuisance_boundary,
        objective=full.objective,
    )


def source_covariance_components(
    qbase: Array,
    contrast: Array,
    family: str,
) -> list[Array]:
    """Scale-aware covariance components for empirical summary analyses."""

    qbase = _sym(np.asarray(qbase, dtype=float))
    dimension = qbase.shape[0]
    if family == "source_single":
        return [qbase]
    if family == "source_all":
        return [
            np.diag(np.eye(dimension)[index] * qbase[index, index])
            for index in range(dimension)
            if qbase[index, index] > 0
        ]
    if family == "rotated_fitted":
        geometry = target_geometry(qbase, contrast)
        target_projector = geometry.target_basis.T @ geometry.target_basis
        nuisance_projector = geometry.nuisance_basis.T @ geometry.nuisance_basis
        return [
            _sym(geometry.colouring @ target_projector @ geometry.colouring.T),
            _sym(geometry.colouring @ nuisance_projector @ geometry.colouring.T),
        ]
    if family == "target_direct":
        return [_sym(np.asarray(contrast) @ qbase @ np.asarray(contrast).T)]
    raise ValueError(f"Unknown covariance family: {family}")


def fixed_effect_prediction_covariance(
    coefficient_covariance: Array,
    predictor_row: Array,
    contrast: Array,
    source_dimension: int,
) -> Array:
    """Project uncertainty in vec(B) to a held-out target mean."""

    design_block = np.kron(np.asarray(predictor_row).reshape(1, -1), np.eye(source_dimension))
    target_map = np.asarray(contrast) @ design_block
    return _sym(target_map @ coefficient_covariance @ target_map.T)


def target_focused_loso_heteroscedastic(
    source_observations: Array,
    source_observation_covariances: Array,
    design: Array,
    contrast: Array,
    qbase: Array,
    workflows: Iterable[str] = (
        "source_single",
        "source_all",
        "target_direct",
        "rotated_fitted",
    ),
) -> LosoResult:
    """Target-focused LOSO with full heteroscedastic first-level covariance."""

    source_observations = np.asarray(source_observations, dtype=float)
    source_observation_covariances = np.asarray(source_observation_covariances, dtype=float)
    design = np.asarray(design, dtype=float)
    contrast = np.atleast_2d(np.asarray(contrast, dtype=float))
    qbase = np.asarray(qbase, dtype=float)
    n_subjects, source_dimension = source_observations.shape
    target_dimension = contrast.shape[0]
    target_observations = source_observations @ contrast.T
    target_covariances = np.stack(
        [contrast @ covariance @ contrast.T for covariance in source_observation_covariances]
    )
    target_qbase = contrast @ qbase @ contrast.T
    workflow_tuple = tuple(workflows)
    pointwise = np.full((n_subjects, len(workflow_tuple)), np.nan)

    for held_out in range(n_subjects):
        training = np.ones(n_subjects, dtype=bool)
        training[held_out] = False
        for workflow_index, workflow in enumerate(workflow_tuple):
            if workflow == "target_direct":
                components = source_covariance_components(
                    target_qbase, np.eye(target_dimension), "target_direct"
                )
                fit = fit_heteroscedastic_reml(
                    target_observations[training],
                    design[training],
                    target_covariances[training],
                    components,
                )
                predictive_mean = design[held_out] @ fit.coefficients
                fixed_covariance = fixed_effect_prediction_covariance(
                    fit.coefficient_covariance,
                    design[held_out],
                    np.eye(target_dimension),
                    target_dimension,
                )
                predictive_covariance = (
                    target_covariances[held_out]
                    + fit.random_covariance
                    + fixed_covariance
                )
            else:
                components = source_covariance_components(qbase, contrast, workflow)
                fit = fit_heteroscedastic_reml(
                    source_observations[training],
                    design[training],
                    source_observation_covariances[training],
                    components,
                )
                source_mean = design[held_out] @ fit.coefficients
                predictive_mean = contrast @ source_mean
                fixed_covariance = fixed_effect_prediction_covariance(
                    fit.coefficient_covariance,
                    design[held_out],
                    contrast,
                    source_dimension,
                )
                predictive_covariance = (
                    target_covariances[held_out]
                    + contrast @ fit.random_covariance @ contrast.T
                    + fixed_covariance
                )
            pointwise[held_out, workflow_index] = multivariate_normal_logpdf(
                target_observations[held_out], predictive_mean, predictive_covariance
            )

    elpd = pointwise.sum(axis=0)
    pairwise_delta = elpd[:, None] - elpd[None, :]
    pairwise_se = np.zeros_like(pairwise_delta)
    for first in range(len(workflow_tuple)):
        for second in range(len(workflow_tuple)):
            difference = pointwise[:, first] - pointwise[:, second]
            pairwise_se[first, second] = math.sqrt(
                n_subjects * float(np.var(difference, ddof=1))
            )
    best = int(np.argmax(elpd))
    compatible = pairwise_delta[best] <= 2.0 * pairwise_se[best]
    compatible[best] = True
    return LosoResult(
        workflows=workflow_tuple,
        pointwise_log_density=pointwise,
        elpd=elpd,
        pairwise_delta=pairwise_delta,
        pairwise_se=pairwise_se,
        compatible=compatible,
        best_index=best,
    )


def target_focused_loso_common(
    source_observations: Array,
    design: Array,
    source_observation_covariance: Array,
    contrast: Array,
    workflows: Iterable[str] = (
        "source_single",
        "source_all",
        "target_direct",
        "rotated_fitted",
    ),
) -> LosoResult:
    """LOSO plug-in prediction of the same target under every specification."""

    source_observations = np.asarray(source_observations, dtype=float)
    design = np.asarray(design, dtype=float)
    source_observation_covariance = _sym(np.asarray(source_observation_covariance, dtype=float))
    contrast = np.atleast_2d(np.asarray(contrast, dtype=float))
    workflow_tuple = tuple(workflows)
    n_subjects, full_dimension = source_observations.shape
    target_observations = source_observations @ contrast.T
    target_observation_covariance = _sym(
        contrast @ source_observation_covariance @ contrast.T
    )
    pointwise = np.full((n_subjects, len(workflow_tuple)), np.nan)

    for held_out in range(n_subjects):
        training = np.ones(n_subjects, dtype=bool)
        training[held_out] = False
        x_train = design[training]
        x_test = design[held_out]
        for workflow_index, workflow in enumerate(workflow_tuple):
            if workflow == "target_direct":
                fit = fit_common_covariance_reml(
                    target_observations[training],
                    x_train,
                    target_observation_covariance,
                    covariance_components(full_dimension, contrast, workflow),
                )
                predictive_mean = x_test @ fit.coefficients
                leverage = float(x_test @ fit.coefficient_row_covariance @ x_test)
                predictive_covariance = (
                    target_observation_covariance
                    + fit.random_covariance
                    + leverage * fit.total_covariance
                )
            else:
                fit = fit_common_covariance_reml(
                    source_observations[training],
                    x_train,
                    source_observation_covariance,
                    covariance_components(full_dimension, contrast, workflow),
                )
                predictive_mean = contrast @ (x_test @ fit.coefficients)
                leverage = float(x_test @ fit.coefficient_row_covariance @ x_test)
                predictive_covariance = (
                    target_observation_covariance
                    + contrast @ fit.random_covariance @ contrast.T
                    + leverage * contrast @ fit.total_covariance @ contrast.T
                )
            pointwise[held_out, workflow_index] = multivariate_normal_logpdf(
                target_observations[held_out], predictive_mean, predictive_covariance
            )

    elpd = pointwise.sum(axis=0)
    pairwise_delta = elpd[:, None] - elpd[None, :]
    pairwise_se = np.zeros_like(pairwise_delta)
    for first in range(len(workflow_tuple)):
        for second in range(len(workflow_tuple)):
            differences = pointwise[:, first] - pointwise[:, second]
            pairwise_se[first, second] = math.sqrt(
                n_subjects * float(np.var(differences, ddof=1))
            )
    best = int(np.argmax(elpd))
    compatible = pairwise_delta[best] <= 2.0 * pairwise_se[best]
    compatible[best] = True
    return LosoResult(
        workflows=workflow_tuple,
        pointwise_log_density=pointwise,
        elpd=elpd,
        pairwise_delta=pairwise_delta,
        pairwise_se=pairwise_se,
        compatible=compatible,
        best_index=best,
    )
