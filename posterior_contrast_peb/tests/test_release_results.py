"""Integrity checks for the frozen, manuscript-facing result tables."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_results_manifest() -> None:
    manifest = pd.read_csv(RESULTS / "manifest.csv", keep_default_na=False)
    assert not manifest.empty
    assert not manifest["relative_path"].duplicated().any()
    assert "manifest.csv" not in set(manifest["relative_path"])

    expected_paths = {
        path.relative_to(RESULTS).as_posix()
        for path in RESULTS.rglob("*")
        if path.is_file() and path.name != "manifest.csv"
    }
    assert set(manifest["relative_path"]) == expected_paths

    for _, row in manifest.iterrows():
        path = RESULTS / row["relative_path"]
        payload = path.read_bytes()
        assert len(payload) == int(row["bytes"])
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                data_rows = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
            assert data_rows == int(row["rows"])
        else:
            assert row["rows"] == ""


def test_data_s1_posterior_anchored_grid() -> None:
    table = pd.read_csv(RESULTS / "data_s1" / "posterior_anchored_main_grid.csv")
    assert len(table) == 96
    assert set(table["workflow"]) == {
        "source_single", "source_all", "target_direct", "rotated_fitted"
    }
    assert set(table["replications"]) == {500}
    assert table["coverage"].between(0.0, 1.0).all()
    assert table["coverage_wilson_lower"].between(0.0, 1.0).all()
    assert table["coverage_wilson_upper"].between(0.0, 1.0).all()
    assert (table["coverage_wilson_lower"] <= table["coverage"]).all()
    assert (table["coverage"] <= table["coverage_wilson_upper"]).all()


def assert_exact_coverage_metadata(table: pd.DataFrame) -> None:
    prefixes = [
        column[:-5]
        for column in table.columns
        if column.endswith("_coverage_mean")
        and f"{column[:-5]}_lower95" in table.columns
    ]
    assert prefixes
    for prefix in prefixes:
        mean = f"{prefix}_mean"
        se = f"{prefix}_se"
        lower = f"{prefix}_lower95"
        upper = f"{prefix}_upper95"
        method = f"{prefix}_interval_method"
        counts = f"{prefix}_outcome_counts"
        assert method in table.columns
        assert counts in table.columns

        observed = table[mean].notna()
        subset = table.loc[observed]
        if subset.empty:
            continue
        assert subset[lower].between(0.0, 1.0).all()
        assert subset[upper].between(0.0, 1.0).all()
        assert (subset[lower] <= subset[mean]).all()
        assert (subset[mean] <= subset[upper]).all()
        expected_method = subset["target_dimension"].map({
            1: "central_exact_binomial_clopper_pearson",
            2: "central_exact_trinomial_test_inversion",
        })
        assert subset[method].equals(expected_method)

        for _, row in subset.iterrows():
            recovered = tuple(int(value) for value in row[counts].split(","))
            assert sum(recovered) == row["replications"]
            if row["target_dimension"] == 1:
                assert len(recovered) == 2
                recovered_mean = recovered[1] / sum(recovered)
                values = np.repeat((0.0, 1.0), recovered)
            else:
                assert len(recovered) == 3
                recovered_mean = (recovered[1] + 2 * recovered[2]) / (
                    2 * sum(recovered)
                )
                values = np.repeat((0.0, 0.5, 1.0), recovered)
            recovered_se = np.std(values, ddof=1) / np.sqrt(sum(recovered))
            assert np.isclose(recovered_mean, row[mean])
            assert np.isclose(recovered_se, row[se])


def test_data_s2_native_spm_grid() -> None:
    information = pd.read_csv(
        RESULTS / "data_s2" / "spm12_information_audit_summary.csv"
    )
    calibration = pd.read_csv(
        RESULTS / "data_s2" / "spm12_hyperparameter_calibration_summary.csv"
    )
    assert len(information) == 30
    assert len(calibration) == 18
    assert set(information["replications"]) == {30}
    assert set(calibration["replications"]) == {50}
    assert set(information["local_target_rank"]) == {1, 2}
    assert_exact_coverage_metadata(information)
    assert_exact_coverage_metadata(calibration)

    zero = calibration[
        (calibration["scenario"] == "target_sparse")
        & (calibration["specification"] == "source_single")
    ]
    assert zero["mean_target_variance_coverage_mean"].eq(0.0).all()
    assert np.allclose(
        zero["mean_target_variance_coverage_upper95"],
        0.07112173645,
        atol=2e-10,
    )


def test_data_s3_nested_prediction() -> None:
    pointwise = pd.read_csv(RESULTS / "data_s3" / "nested_pointwise.csv")
    weights = pd.read_csv(RESULTS / "data_s3" / "stacking_weights.csv")
    assert set(pointwise["target_dimension"]) == {1, 2}
    assert pointwise["held_out"].nunique() == 76
    assert len(pointwise) == 76 * 2 * 8
    assert set(pointwise["method"]) == {
        "source_single", "source_all", "target_direct", "rotated_fitted",
        "hard_inner_selection", "equal_average", "regularized_stacking",
        "pseudo_bma_plus",
    }
    effective = weights.groupby("target_dimension")["effective_candidate_count"].first()
    assert np.isclose(effective.loc[1], 3.750260284072522)
    assert np.isclose(effective.loc[2], 3.4697341905628822)


def test_data_s4_empirical_information_audit() -> None:
    audit = pd.read_csv(
        RESULTS / "data_s4" / "empirical_target_information_audit_all_datasets.csv"
    )
    assert len(audit) == 20
    assert audit.groupby(["dataset", "target"]).size().eq(5).all()
    assert audit["total_target_information_fraction"].between(0.0, 1.0).all()
    assert audit["visible_direction_information_fraction"].between(0.0, 1.0).all()
    assert np.isclose(
        audit["minimum_retained_generalized_eigenvalue"].min(),
        0.0412019067640586,
    )
    assert audit["maximum_absolute_discarded_generalized_eigenvalue"].max() < 1.3e-16
    assert audit["rank_invariant_1e_9_to_0_04"].eq(1).all()

    ordering = pd.read_csv(
        RESULTS / "data_s4" / "empirical_parameter_ordering.csv"
    )
    assert ordering.groupby("dataset").size().to_dict() == {
        "FacesData": 18,
        "FingerData": 18,
        "GeometryData": 12,
    }
    assert not ordering.duplicated(["dataset", "source_index"]).any()
    assert ordering["parameter_name"].str.match(r"^[ABC]\(").all()

    targets = pd.read_csv(
        RESULTS / "data_s4" / "empirical_target_matrices.csv"
    )
    dimensions = ordering.groupby("dataset").size()
    assert targets["source_index"].le(targets["dataset"].map(dimensions)).all()

    fit_keys = ["dataset", "target", "specification"]
    projections = pd.read_csv(
        RESULTS / "data_s4" / "empirical_fitted_target_projections.csv"
    )
    assert projections.groupby(fit_keys).ngroups == 20
    assert not projections.duplicated(
        fit_keys + ["target_row", "coordinate_index"]
    ).any()
    assert projections["target_row"].between(
        1, projections["target_dimension"]
    ).all()
    assert projections["coordinate_index"].between(
        1, projections["coordinate_dimension"]
    ).all()

    components = pd.read_csv(
        RESULTS / "data_s4" / "empirical_precision_components_long.csv"
    )
    baseline = pd.read_csv(
        RESULTS / "data_s4" / "empirical_baseline_precision_long.csv"
    )
    assert components.groupby(fit_keys).ngroups == 20
    assert baseline.groupby(fit_keys).ngroups == 20
    assert components["component_scale"].gt(0).all()
    assert np.allclose(
        components["weighted_value"],
        components["component_scale"] * components["base_value"],
    )
    assert components["row_index"].between(
        1, components["coordinate_dimension"]
    ).all()
    assert components["column_index"].between(
        1, components["coordinate_dimension"]
    ).all()

    released_counts = (
        components.groupby(fit_keys)["component_index"].nunique().rename("released")
    )
    expected_counts = audit.set_index(fit_keys)["n_components"].rename("expected")
    joined = pd.concat([released_counts, expected_counts], axis=1)
    assert not joined.isna().any().any()
    assert joined["released"].eq(joined["expected"]).all()

    matrix_keys = fit_keys + ["component_index", "row_index", "column_index"]
    assert not components.duplicated(matrix_keys).any()
    assert not baseline.duplicated(
        fit_keys + ["row_index", "column_index"]
    ).any()
    for _, group in components.groupby(fit_keys + ["component_index"]):
        dimension = int(group["coordinate_dimension"].iloc[0])
        matrix = np.zeros((dimension, dimension))
        matrix[
            group["row_index"].to_numpy(dtype=int) - 1,
            group["column_index"].to_numpy(dtype=int) - 1,
        ] = group["base_value"].to_numpy()
        assert np.allclose(matrix, matrix.T, atol=1e-12)

    def parse_matrix(value: str) -> np.ndarray:
        return np.asarray([
            [float(entry) for entry in row.split(",")]
            for row in str(value).split(";")
        ])

    audit_lookup = audit.set_index(fit_keys)
    for key, group in components.groupby(fit_keys):
        dimension = int(group["coordinate_dimension"].iloc[0])
        precision = np.zeros((dimension, dimension))
        baseline_rows = baseline.set_index(fit_keys).loc[key]
        if isinstance(baseline_rows, pd.Series):
            baseline_rows = baseline_rows.to_frame().T
        precision[
            baseline_rows["row_index"].to_numpy(dtype=int) - 1,
            baseline_rows["column_index"].to_numpy(dtype=int) - 1,
        ] = baseline_rows["value"].to_numpy(dtype=float)
        for _, component in group.groupby("component_index"):
            precision[
                component["row_index"].to_numpy(dtype=int) - 1,
                component["column_index"].to_numpy(dtype=int) - 1,
            ] += component["weighted_value"].to_numpy()
        covariance = np.linalg.inv(precision)
        target_dimension = int(group["target_dimension"].iloc[0])
        projection_rows = projections.set_index(fit_keys).loc[key]
        if isinstance(projection_rows, pd.Series):
            projection_rows = projection_rows.to_frame().T
        projection = np.zeros((target_dimension, dimension))
        projection[
            projection_rows["target_row"].to_numpy(dtype=int) - 1,
            projection_rows["coordinate_index"].to_numpy(dtype=int) - 1,
        ] = projection_rows["weight"].to_numpy(dtype=float)
        reconstructed = projection @ covariance @ projection.T
        expected = parse_matrix(
            audit_lookup.loc[key, "target_random_effects_covariance"]
        )
        assert np.allclose(reconstructed, expected, atol=1e-8)
