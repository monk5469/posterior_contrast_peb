# Data S4 computational provenance

All final Native-SPM empirical audits used MATLAB R2023b
(`23.2.0.2365128`) and SPM12 revision 7771 (13-Jan-2020). The PEB fits used
`beta=16` and `maxit=64`. FacesData and GeometryData used intercept-only
second-level designs. No first-level DCM was reinverted for this release.

## Frozen derived inputs

The subject-level posterior-summary archives are not redistributed. Their
SHA-256 hashes identify the exact frozen inputs used to generate the released
non-identifying summaries.

| Dataset | Source dimension | Input SHA-256 |
|---|---:|---|
| FacesData | 18 | `C5E0113A03F9B34E860553F0664C4A68FB70DA739B2D162E26B7D5E6638D521F` |
| GeometryData | 12 | `659BF1F699A3CC311B72BB2681AA2F91939C5181113565CA372C15DB717CB8C9` |
| FingerData | 18 | `E9197CF73C273A97A15D7FDAA5398E29B9ABBE24012F7C0EFFC4D2B5FE6CB8A7` |

## SPM implementation

| File | SHA-256 |
|---|---|
| `spm_dcm_peb.m` | `4D7A8FE4BFC91AA51B5579E7C5B4D40BF46FD7955B25E1C98ED173BBD457260F` |
| `spm_reml_sc.m` | `1292C4291BD7C1AECF0C5C39D4302E25F1FE9186E6371443ECCD868323C2541A` |

FacesData and GeometryData were refitted under this environment on
2026-09-15, producing five specifications for each dataset. Their final
R2023b summaries were compared with the archived R2025a outputs; group target
means, posterior covariances, target random-effects covariances, generalised-
information eigenvalues, and free energies were identical at the precision
written to the release tables. FingerData used the same MATLAB and SPM12
release and contributed five specifications for each of its two target
definitions.

## Released component representation

`empirical_precision_components_long.csv` is generated from the frozen fitted
PEB objects by
`simulations/spm12/export_empirical_release_components.m`. Each row is a
nonzero entry of one fitted precision-component matrix and includes its fitted
log scale, scale `exp(Eh)`, base value, and weighted value.
`empirical_baseline_precision_long.csv` records the residual baseline term
`P0 = inv(PEB.Ce) - sum_k exp(Eh_k) Q_k`, using the same stabilised inverse as
the released audit. The complete fitted precision is reconstructed by adding
this baseline term to the sum of `weighted_value` over component indices.
Coordinate systems are explicitly labelled as source, target, or rotated
target-complement coordinates.

`empirical_parameter_ordering.csv` gives the dense source-coordinate ordering.
`empirical_target_matrices.csv` gives the sparse nonzero representation of
each declared target matrix. `empirical_fitted_target_projections.csv` records
the projection actually used by each fitted source, rotated, or direct model,
so every released fitted precision can be inverted and projected back to its
reported target covariance. Unlisted matrix entries and target weights are
zero. Because the residual baseline term is defined from the fitted `PEB.Ce`,
this reconstruction verifies release completeness and numerical consistency;
it is not an independent refit of the empirical PEB models.
