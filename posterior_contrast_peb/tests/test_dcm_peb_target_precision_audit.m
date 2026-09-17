function test_dcm_peb_target_precision_audit
%TEST_DCM_PEB_TARGET_PRECISION_AUDIT Analytic and finite-difference checks.

Q = {diag([1 0 0]), diag([0 1 0]), diag([1 1 0])};
g = log([1.2; 0.7; 0.9]);
P0 = [2.0 0.3 0.1; 0.3 1.8 0.2; 0.1 0.2 1.5];
P = P0;
for k = 1:numel(Q)
    P = P + exp(g(k)) * Q{k};
end

PEB = struct();
PEB.M.Q = Q;
PEB.Eh = g;
PEB.Ce = inv(P);
PEB.Ch = eye(numel(Q));
L = [1 0 0; 0 1 0];

audit = dcm_peb_target_precision_audit(PEB, L);
assert(audit.local_target_rank == 2);
assert(audit.local_target_rank <= audit.target_symmetric_dimension);
assert(audit.local_nullity == 1);
assert(abs(audit.target_rank_fraction - 2/3) < 1e-10);
assert(abs(audit.component_redundancy_fraction - 1/3) < 1e-10);
assert(audit.ce_reconstruction_relative_error < 1e-10);
assert(audit.max_finite_difference_relative_error < 1e-8);
assert(numel(audit.selected_component_indices) == 2);
assert(audit.has_hyperparameter_covariance);
assert(abs(audit.null_hyperparameter_uncertainty_fraction - 1/3) < 1e-10);
assert(isequal(size(audit.target_covariance_delta_uncertainty), [3 3]));
assert(isequal(size(audit.target_covariance_delta_standard_error), [2 2]));

% Scalar targets can see at most one local precision combination.
scalar_audit = dcm_peb_target_precision_audit(PEB, [1 0 0]);
assert(scalar_audit.local_target_rank == 1);
assert(scalar_audit.local_nullity == 2);
assert(scalar_audit.target_rank_fraction == 1);
assert(abs(scalar_audit.component_redundancy_fraction - 2/3) < 1e-10);
assert(abs(scalar_audit.null_hyperparameter_uncertainty_fraction - 2/3) < 1e-10);

fprintf('dcm_peb_target_precision_audit: all tests passed.\n');
end
