# Data availability

The repository contains non-identifying, derived numerical summaries used in
the manuscript and supplement. The files are grouped as Data S1-S4 under
`posterior_contrast_peb/results/`.

Raw BIDS data, fMRIPrep derivatives, SPM first-level folders, VOI files, and
subject-level DCM inversion files are not redistributed. The three empirical
examples use publicly available source datasets described in the manuscript;
accession information and original citations are provided there.

The public tables are sufficient to verify the reported aggregate values and
to reproduce manuscript-facing numerical summaries. Re-running empirical fits
requires locally prepared subject-level posterior means, posterior covariance
matrices, parameter names, group labels, and the declared covariate table.

Data S4 includes non-identifying source-parameter orderings, target matrices,
and sparse long-form records of the fitted precision-component dictionaries,
residual baseline precision terms, and fitted target projections. These
derived records permit reconstruction of the reported target random-effects
covariance matrices without redistributing subject-level posterior files.

No direct identifiers, imaging volumes, or internal project logs are included.
