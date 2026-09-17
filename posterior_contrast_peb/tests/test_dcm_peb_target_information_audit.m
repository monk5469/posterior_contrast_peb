function test_dcm_peb_target_information_audit
%TEST_DCM_PEB_TARGET_INFORMATION_AUDIT Analytic information-fraction checks.

D = 4;
N = 10;
Ce = 0.2 * eye(D);
subject_covariances = repmat({0.1 * eye(D)}, N, 1);
L = [1 0 0 0];

single = synthetic_peb({eye(D)}, 0, Ce);
single_audit = dcm_peb_target_information_audit( ...
    single, L, subject_covariances);
assert(abs(single_audit.total_target_information_fraction - 1 / D) < 1e-10);
assert(abs(single_audit.visible_direction_information_fraction - 1 / D) < 1e-10);
assert(single_audit.full_information_rank == 1);
assert(single_audit.target_information_rank == 1);
assert(single_audit.generalized_rank_tolerance == 1e-8);
assert(single_audit.eigenvalue_clip_tolerance == 1e-8);
assert(single_audit.full_support_relative_tolerance == 1e-8);
assert(single_audit.full_support_absolute_tolerance == 1e-12);
assert(single_audit.minimum_retained_generalized_eigenvalue > 0);
assert(isnan(single_audit.maximum_absolute_discarded_generalized_eigenvalue));

Q = cell(D, 1);
for k = 1:D
    Q{k} = zeros(D);
    Q{k}(k, k) = 1;
end
all_components = synthetic_peb(Q, zeros(D, 1), Ce);
all_audit = dcm_peb_target_information_audit( ...
    all_components, L, subject_covariances);
assert(abs(all_audit.total_target_information_fraction - 1 / D) < 1e-10);
assert(abs(all_audit.visible_direction_information_fraction - 1) < 1e-10);
assert(all_audit.full_information_rank == D);
assert(all_audit.target_information_rank == 1);
assert(min(all_audit.generalized_information_eigenvalues) >= -1e-10);
assert(max(all_audit.generalized_information_eigenvalues) <= 1 + 1e-10);
assert(all_audit.minimum_conditional_information_eigenvalue >= -1e-10);
assert(all_audit.maximum_absolute_discarded_generalized_eigenvalue < 1e-10);

legacy_audit = dcm_peb_target_information_audit( ...
    all_components, L, subject_covariances, ...
    'relative_tolerance', 1e-7, 'absolute_tolerance', 1e-11, ...
    'clip_tolerance', 1e-7);
assert(legacy_audit.full_support_relative_tolerance == 1e-7);
assert(legacy_audit.full_support_absolute_tolerance == 1e-11);
assert(legacy_audit.generalized_rank_tolerance == 1e-7);
assert(legacy_audit.eigenvalue_clip_tolerance == 1e-7);

direct = synthetic_peb({1}, 0, 0.2);
direct_audit = dcm_peb_target_information_audit(direct, 1, repmat({0.1}, N, 1));
assert(abs(direct_audit.total_target_information_fraction - 1) < 1e-10);
assert(abs(direct_audit.visible_direction_information_fraction - 1) < 1e-10);

fprintf('dcm_peb_target_information_audit: all tests passed.\n');
end

function PEB = synthetic_peb(Q, g, Ce)
PEB = struct();
PEB.M.Q = Q;
PEB.Eh = g;
PEB.Ch = eye(numel(Q));
PEB.Ce = Ce;
end
