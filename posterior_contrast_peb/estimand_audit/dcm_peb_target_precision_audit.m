function audit = dcm_peb_target_precision_audit(PEB, L, varargin)
%DCM_PEB_TARGET_PRECISION_AUDIT Audit target-visible precision directions.
%
% audit = dcm_peb_target_precision_audit(PEB,L) evaluates the local map
%
%   g -> L * inv(P0 + sum_k exp(g_k) Q_k) * L'
%
% at PEB.Eh. P0 is recovered from PEB.Ce and the fitted precision
% components, avoiding assumptions about whether the supplied component
% dictionary partitions the baseline precision.
%
% This function diagnoses local target visibility. Selected component
% indices are a column-subset diagnostic only; removing other components
% does not in general preserve the full source-space likelihood.

opts = parse_options(varargin{:});
validateattributes(L, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'});

if ~isstruct(PEB) || ~isfield(PEB, 'M') || ~isfield(PEB.M, 'Q') || ...
        ~isfield(PEB, 'Eh') || ~isfield(PEB, 'Ce')
    error('PEB must contain M.Q, Eh, and Ce.');
end

Q = PEB.M.Q;
if ~iscell(Q) || isempty(Q)
    error('PEB.M.Q must be a non-empty cell array.');
end
g = full(PEB.Eh(:));
if numel(g) ~= numel(Q)
    error('numel(PEB.Eh) must equal numel(PEB.M.Q).');
end

Cfit = symmetrize(full(PEB.Ce));
D = size(Cfit, 1);
if size(Cfit, 2) ~= D || size(L, 2) ~= D
    error('L and PEB.Ce have incompatible dimensions.');
end
if rank(L, opts.matrix_tolerance) ~= size(L, 1)
    error('L must have full row rank.');
end

for k = 1:numel(Q)
    Q{k} = symmetrize(full(Q{k}));
    if ~isequal(size(Q{k}), [D D])
        error('Precision component %d has the wrong size.', k);
    end
end

Pfit = stable_inverse(Cfit);
weighted_precision = zeros(D);
for k = 1:numel(Q)
    weighted_precision = weighted_precision + exp(g(k)) * Q{k};
end
P0 = symmetrize(Pfit - weighted_precision);

P = P0 + weighted_precision;
C = symmetrize(stable_inverse(P));
A = symmetrize(L * C * L');
reconstruction_error = norm(C - Cfit, 'fro') / max(norm(Cfit, 'fro'), eps);

target_dimension = size(L, 1);
vech_dimension = target_dimension * (target_dimension + 1) / 2;
J = zeros(vech_dimension, numel(Q));
component_norm = zeros(numel(Q), 1);
for k = 1:numel(Q)
    derivative = -exp(g(k)) * L * C * Q{k} * C * L';
    J(:, k) = symmetric_vector(derivative);
    component_norm(k) = norm(J(:, k));
end

[U, S, V] = svd(J, 'econ');
singular_values = diag(S);
if isempty(singular_values) || singular_values(1) == 0
    rank_tolerance = opts.rank_tolerance;
    local_rank = 0;
else
    if isempty(opts.rank_tolerance)
        rank_tolerance = max(opts.absolute_rank_tolerance, ...
            opts.relative_rank_tolerance * singular_values(1));
    else
        rank_tolerance = opts.rank_tolerance;
    end
    local_rank = sum(singular_values > rank_tolerance);
end

if local_rank == 0
    condition_number = Inf;
    selected_indices = zeros(1, 0);
else
    condition_number = singular_values(1) / singular_values(local_rank);
    [~, ~, permutation] = qr(J, 'vector');
    selected_indices = permutation(1:local_rank);
end

null_basis = null(J, opts.matrix_tolerance);

has_hyperparameter_covariance = isfield(PEB, 'Ch') && ...
    isequal(size(PEB.Ch), [numel(Q) numel(Q)]);
hyperparameter_covariance = [];
visible_hyperparameter_uncertainty = NaN;
null_hyperparameter_uncertainty = NaN;
null_hyperparameter_uncertainty_fraction = NaN;
target_covariance_delta_uncertainty = [];
target_covariance_delta_standard_error = [];
target_covariance_delta_interval95_lower = [];
target_covariance_delta_interval95_upper = [];
if has_hyperparameter_covariance
    hyperparameter_covariance = symmetrize(full(PEB.Ch));
    if local_rank > 0
        visible_basis = V(:, 1:local_rank);
        visible_hyperparameter_uncertainty = trace( ...
            visible_basis' * hyperparameter_covariance * visible_basis);
    else
        visible_hyperparameter_uncertainty = 0;
    end
    if isempty(null_basis)
        null_hyperparameter_uncertainty = 0;
    else
        null_hyperparameter_uncertainty = trace( ...
            null_basis' * hyperparameter_covariance * null_basis);
    end
    total_hyperparameter_uncertainty = trace(hyperparameter_covariance);
    if total_hyperparameter_uncertainty > 0
        null_hyperparameter_uncertainty_fraction = ...
            null_hyperparameter_uncertainty / total_hyperparameter_uncertainty;
    end
    target_covariance_delta_uncertainty = symmetrize( ...
        J * hyperparameter_covariance * J');
    standard_error_vector = sqrt(max( ...
        diag(target_covariance_delta_uncertainty), 0));
    target_covariance_delta_standard_error = ...
        symmetric_matrix(standard_error_vector, target_dimension);
    target_covariance_delta_interval95_lower = ...
        A - 1.96 * target_covariance_delta_standard_error;
    target_covariance_delta_interval95_upper = ...
        A + 1.96 * target_covariance_delta_standard_error;
end

finite_difference_absolute_errors = nan(numel(Q), 1);
finite_difference_errors = nan(numel(Q), 1);
h = opts.finite_difference_step;
global_sensitivity_scale = max(norm(J, 'fro'), opts.absolute_rank_tolerance);
for k = 1:numel(Q)
    gp = g;
    gm = g;
    gp(k) = gp(k) + h;
    gm(k) = gm(k) - h;
    Ap = target_covariance(P0, Q, gp, L);
    Am = target_covariance(P0, Q, gm, L);
    numeric_derivative = symmetric_vector((Ap - Am) / (2 * h));
    finite_difference_absolute_errors(k) = norm(numeric_derivative - J(:, k));
    finite_difference_errors(k) = finite_difference_absolute_errors(k) / ...
        global_sensitivity_scale;
end

selected = false(numel(Q), 1);
selected(selected_indices) = true;
component_table = table((1:numel(Q))', g, component_norm, selected, ...
    finite_difference_absolute_errors, finite_difference_errors, 'VariableNames', ...
    {'component', 'log_precision', 'target_sensitivity_norm', ...
     'pivot_selected', 'finite_difference_absolute_error', ...
     'finite_difference_global_relative_error'});

audit = struct();
audit.full_dimension = D;
audit.target_dimension = target_dimension;
audit.target_symmetric_dimension = vech_dimension;
audit.n_precision_components = numel(Q);
audit.local_target_rank = local_rank;
audit.local_nullity = numel(Q) - local_rank;
audit.target_rank_fraction = local_rank / vech_dimension;
audit.component_redundancy_fraction = ...
    (numel(Q) - local_rank) / numel(Q);
audit.rank_tolerance = rank_tolerance;
audit.singular_values = singular_values;
audit.condition_number = condition_number;
audit.target_covariance = A;
audit.baseline_precision = P0;
audit.target_jacobian = J;
audit.left_singular_vectors = U;
audit.hyperparameter_singular_vectors = V;
audit.null_basis = null_basis;
audit.has_hyperparameter_covariance = has_hyperparameter_covariance;
audit.hyperparameter_posterior_covariance = hyperparameter_covariance;
audit.visible_hyperparameter_uncertainty = visible_hyperparameter_uncertainty;
audit.null_hyperparameter_uncertainty = null_hyperparameter_uncertainty;
audit.null_hyperparameter_uncertainty_fraction = ...
    null_hyperparameter_uncertainty_fraction;
audit.target_covariance_delta_uncertainty = ...
    target_covariance_delta_uncertainty;
audit.target_covariance_delta_standard_error = ...
    target_covariance_delta_standard_error;
audit.target_covariance_delta_interval95_lower = ...
    target_covariance_delta_interval95_lower;
audit.target_covariance_delta_interval95_upper = ...
    target_covariance_delta_interval95_upper;
audit.selected_component_indices = selected_indices;
audit.component_table = component_table;
audit.ce_reconstruction_relative_error = reconstruction_error;
audit.max_finite_difference_relative_error = max(finite_difference_errors);
end

function opts = parse_options(varargin)
opts = struct('rank_tolerance', [], 'relative_rank_tolerance', 1e-8, ...
    'absolute_rank_tolerance', 1e-12, 'matrix_tolerance', 1e-10, ...
    'finite_difference_step', 1e-5);
if mod(numel(varargin), 2) ~= 0
    error('Options must be supplied as name/value pairs.');
end
for i = 1:2:numel(varargin)
    name = lower(char(varargin{i}));
    if ~isfield(opts, name)
        error('Unknown option: %s', name);
    end
    opts.(name) = varargin{i + 1};
end
end

function A = target_covariance(P0, Q, g, L)
P = P0;
for k = 1:numel(Q)
    P = P + exp(g(k)) * Q{k};
end
C = stable_inverse(symmetrize(P));
A = symmetrize(L * C * L');
end

function vector = symmetric_vector(matrix)
matrix = symmetrize(matrix);
n = size(matrix, 1);
vector = zeros(n * (n + 1) / 2, 1);
index = 1;
for column = 1:n
    for row = 1:column
        value = matrix(row, column);
        if row ~= column
            value = sqrt(2) * value;
        end
        vector(index) = value;
        index = index + 1;
    end
end
end

function matrix = symmetric_matrix(vector, n)
matrix = zeros(n);
index = 1;
for column = 1:n
    for row = 1:column
        value = vector(index);
        if row ~= column
            value = value / sqrt(2);
        end
        matrix(row, column) = value;
        matrix(column, row) = value;
        index = index + 1;
    end
end
end

function matrix = stable_inverse(matrix)
matrix = symmetrize(matrix);
[V, D] = eig(matrix);
values = real(diag(D));
floor_value = max(max(abs(values)) * 1e-12, eps);
values(values < floor_value) = floor_value;
matrix = symmetrize(V * diag(1 ./ values) * V');
end

function matrix = symmetrize(matrix)
matrix = (matrix + matrix') / 2;
end
