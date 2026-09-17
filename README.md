# Estimand-Relative Auditing and Specification Uncertainty in PEB

This repository contains the public analysis code and non-identifying derived
results for:

**Estimand-Relative Auditing and Specification Uncertainty in Parametric
Empirical Bayes for Repeated-Measures Dynamic Causal Modelling**

The implementation addresses inference on prespecified linear targets of
subject-level DCM parameters. It provides:

- joint propagation of first-level posterior means and covariances into a
  target space;
- estimand-relative audits of fitted PEB precision-component dictionaries;
- propagation of PEB hyperparameter uncertainty to target random-effects
  covariance;
- predictive comparison and averaging of prespecified covariance-component
  specifications; and
- native-SPM validation and target-family BMR/BMS checks.

For a subject-level Gaussian posterior,

```text
theta_s | y_s ~ N(mu_s, Sigma_s)
m_s = L mu_s
V_s = L Sigma_s L'
```

where `L` encodes the scientific target. The audit then asks how much of the
fitted precision-component information is available for that target and how
uncertainty in the fitted components propagates to target-level quantities.

## Repository layout

```text
posterior_contrast_peb/
  core/                  posterior projection and PEB wrapper functions
  estimand_audit/        target-relative precision and information audits
  analysis/              Gaussian/ReML covariance-specification utilities
  simulations/
    posterior_anchored/  posterior-anchored finite-sample experiment
    prediction/          predictive averaging and nested resampling
    spm12/               native-SPM validation and BMR/BMS checks
  results/
    data_s1/             posterior-anchored variance-allocation summaries
    data_s2/             SPM12 information and calibration summaries
    data_s3/             predictive comparison and averaging summaries
    data_s4/             empirical target-information audits
    secondary_validation/
  tests/                 numerical contracts and release-data checks
  docs/                  data and reproducibility notes
```

No manuscript drafts, figures, plotting code, internal audit notes, raw
neuroimaging data, or subject-level DCM inversion files are included.

## Requirements

- MATLAB R2023b
- SPM12 revision 7771
- Python 3.10 or later
- Python packages listed in `requirements.txt`

## Verification

Install the Python requirements and run:

```bash
python -m pytest posterior_contrast_peb/tests/test_specification_averaging.py \
  posterior_contrast_peb/tests/test_release_results.py
```

For the MATLAB numerical contracts:

```matlab
addpath(genpath('posterior_contrast_peb'));
run('posterior_contrast_peb/tests/test_dcm_peb_target_precision_audit.m');
run('posterior_contrast_peb/tests/test_dcm_peb_target_information_audit.m');
run('posterior_contrast_peb/tests/test_dcm_peb_target_precision_posterior.m');
```

The SPM-dependent wrapper test additionally requires SPM12 on the MATLAB path:

```matlab
addpath('/path/to/spm12');
addpath(genpath('posterior_contrast_peb'));
run('posterior_contrast_peb/tests/test_spm_dcm_peb_contrast_wrapper.m');
```

The native-SPM simulation also accepts an `SPM12_DIR` environment variable
when SPM12 is not already on the MATLAB path.

## Empirical inputs

Public derived summaries are provided under `results/`. Raw BIDS data,
fMRIPrep derivatives, SPM first-level folders, VOI files, and subject-level
DCM inversion files are not redistributed. The empirical runners therefore
accept an explicit derived posterior-summary file and covariate table.

See `posterior_contrast_peb/docs/data_availability.md` and
`posterior_contrast_peb/docs/reproducibility.md`.

## Authors

Haifeng Wu, Jia Tang, and Yu Zeng. Correspondence:
`whf5469@gmail.com`.

## License

BSD 3-Clause License. See `LICENSE`.
