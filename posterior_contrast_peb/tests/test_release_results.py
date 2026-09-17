"""Integrity checks for the frozen, manuscript-facing result tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_data_s1_posterior_anchored_grid() -> None:
    table = pd.read_csv(RESULTS / "data_s1" / "posterior_anchored_main_grid.csv")
    assert len(table) == 96
    assert set(table["workflow"]) == {
        "source_single", "source_all", "target_direct", "rotated_fitted"
    }
    assert set(table["replications"]) == {500}
    assert table["coverage"].between(0.0, 1.0).all()


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
