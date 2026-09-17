function test_dcm_peb_target_precision_posterior
%TEST_DCM_PEB_TARGET_PRECISION_POSTERIOR Sampling and reproducibility checks.

PEB = struct();
PEB.M.Q = {1};
PEB.Eh = 0;
PEB.Ch = 0.04;
PEB.Ce = 1 / 3;
L = 1;

first = dcm_peb_target_precision_posterior(PEB, L, ...
    'n_draws', 2000, 'seed', 42, 'return_draws', true);
second = dcm_peb_target_precision_posterior(PEB, L, ...
    'n_draws', 2000, 'seed', 42, 'return_draws', true);

assert(isequal(first.covariance_draws, second.covariance_draws));
assert(all(first.covariance_draws(:) > 0));
assert(first.interval_lower < first.posterior_median);
assert(first.posterior_median < first.interval_upper);
assert(first.interval_lower > 0);
assert(abs(first.plugin_target_covariance - 1 / 3) < 1e-12);
assert(first.audit.max_finite_difference_relative_error < 1e-8);

fprintf('dcm_peb_target_precision_posterior: all tests passed.\n');
end
