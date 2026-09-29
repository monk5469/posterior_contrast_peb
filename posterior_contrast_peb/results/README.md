# Released numerical results

This directory contains the non-identifying, manuscript-facing Data S1-S4
tables and secondary validation results. It does not contain manuscript
drafts, figures, plotting code, internal audit notes, raw imaging data, or
subject-level DCM inversion files.

## Data S1

- `posterior_anchored_main_grid.csv`: 500-replication variance-allocation grid.
- `posterior_anchored_covariance_scale_sensitivity.csv`: first-level
  covariance-scale sensitivity analysis.

Coverage intervals in these tables are Wilson intervals.

## Data S2

- `spm12_information_audit_summary.csv`: 30-replication structural-information
  validation grid.
- `spm12_hyperparameter_calibration_summary.csv`: 50-replication nonlinear
  hyperparameter-propagation calibration grid.

Every non-missing coverage family reports its exact interval method and the
underlying replication-level outcome counts. Scalar targets use exact binomial
intervals; two-dimensional targets use paired exact trinomial intervals.
Outcome-count tuples are ordered as `(not covered, covered)` for scalar
targets and `(neither covered, one covered, both covered)` for paired targets.

## Data S3

Data S3 contains independent prediction summaries, FingerData nested
cross-validation outputs, final regularised-stacking weights, candidate
posterior summaries, and the final finite-Gaussian-mixture summaries. The
stacking-weight intervals are 95% bootstrap percentile intervals. FingerData
target ordering follows the labels stored in `posterior_summary.csv`.

## Data S4

Data S4 contains the 20 empirical dataset-target-specification audits,
generalised-information threshold checks and complete spectra, target-
covariance posterior summaries, full source-parameter orderings, sparse target
matrices, realised fitted precision components, residual baseline precision
terms, fitted target projections, and computational provenance.

The component table is a sparse long representation. Unlisted matrix entries
are zero. The target-matrix and fitted-projection tables likewise store only
nonzero weights. The fitted projections make the source, rotated, and direct
component records independently reconstructable when combined with the
released residual baseline precision terms.

## Integrity

`manifest.csv` records file sizes, row counts where applicable, and SHA-256
hashes. Release contracts are checked by `tests/test_release_results.py`.
