function information = dcm_peb_target_information_audit(PEB, L, subject_covariances, varargin)
%DCM_PEB_TARGET_INFORMATION_AUDIT Audit where precision information arises.
%
% Compares local expected Gaussian Fisher information for the fitted
% precision hyperparameters before and after the prespecified projection L.
% The calculation is conditional on the fitted group mean. It diagnoses
% reliance on non-target coordinates; it is not a replacement for the
% ReML Hessian or proof that pooling across coordinates is incorrect.

opts = parse_options(varargin{:});
audit = dcm_peb_target_precision_audit(PEB, L);
subject_covariances = normalize_covariances(subject_covariances, ...
    audit.full_dimension);

Q = PEB.M.Q;
g = full(PEB.Eh(:));
C = symmetrize(full(PEB.Ce));
K = numel(Q);
derivatives = cell(K, 1);
target_derivatives = cell(K, 1);
for k = 1:K
    derivatives{k} = symmetrize(-exp(g(k)) * C * full(Q{k}) * C);
    target_derivatives{k} = symmetrize(L * derivatives{k} * L');
end

full_information = zeros(K);
target_information = zeros(K);
for subject = 1:numel(subject_covariances)
    total_covariance = symmetrize(subject_covariances{subject} + C);
    target_covariance = symmetrize(L * total_covariance * L');
    full_precision = stable_inverse(total_covariance);
    target_precision = stable_inverse(target_covariance);
    full_scores = cell(K, 1);
    target_scores = cell(K, 1);
    for k = 1:K
        full_scores{k} = full_precision * derivatives{k};
        target_scores{k} = target_precision * target_derivatives{k};
    end
    for k = 1:K
        for j = k:K
            full_value = 0.5 * trace(full_scores{k} * full_scores{j});
            target_value = 0.5 * trace(target_scores{k} * target_scores{j});
            full_information(k, j) = full_information(k, j) + full_value;
            target_information(k, j) = target_information(k, j) + target_value;
            if j ~= k
                full_information(j, k) = full_information(j, k) + full_value;
                target_information(j, k) = target_information(j, k) + target_value;
            end
        end
    end
end
full_information = symmetrize(full_information);
target_information = symmetrize(target_information);

[U, D] = eig(full_information);
full_eigenvalues = real(diag(D));
maximum = max(max(abs(full_eigenvalues)), eps);
tolerance = max(opts.full_support_absolute_tolerance, ...
    opts.full_support_relative_tolerance * maximum);
retained = full_eigenvalues > tolerance;
full_rank = sum(retained);
if full_rank == 0
    whitened_target_information = zeros(0);
    raw_generalized_eigenvalues = zeros(0, 1);
    generalized_eigenvalues = zeros(0, 1);
else
    basis = real(U(:, retained));
    scales = full_eigenvalues(retained);
    inverse_root = diag(1 ./ sqrt(scales));
    whitened_target_information = symmetrize(inverse_root * ...
        (basis' * target_information * basis) * inverse_root);
    raw_generalized_eigenvalues = sort( ...
        real(eig(whitened_target_information)), 'descend');
    generalized_eigenvalues = raw_generalized_eigenvalues;
    generalized_eigenvalues(abs(generalized_eigenvalues) < ...
        opts.eigenvalue_clip_tolerance) = 0;
    generalized_eigenvalues(abs(generalized_eigenvalues - 1) < ...
        opts.eigenvalue_clip_tolerance) = 1;
end
positive = raw_generalized_eigenvalues > opts.generalized_rank_tolerance;
target_rank = sum(positive);
if target_rank == 0
    minimum_retained_generalized_eigenvalue = NaN;
else
    minimum_retained_generalized_eigenvalue = min( ...
        raw_generalized_eigenvalues(positive));
end
discarded = raw_generalized_eigenvalues(~positive);
if isempty(discarded)
    maximum_absolute_discarded_generalized_eigenvalue = NaN;
else
    maximum_absolute_discarded_generalized_eigenvalue = max(abs(discarded));
end
if full_rank == 0
    total_target_information_fraction = NaN;
else
    total_target_information_fraction = trace(whitened_target_information) / full_rank;
end
if target_rank == 0
    visible_direction_information_fraction = 0;
else
    visible_direction_information_fraction = mean( ...
        generalized_eigenvalues(1:target_rank));
end

conditional_information = symmetrize(full_information - target_information);
minimum_conditional_eigenvalue = min(real(eig(conditional_information)));

information = struct();
information.full_dimension = audit.full_dimension;
information.target_dimension = audit.target_dimension;
information.n_precision_components = K;
information.n_subjects = numel(subject_covariances);
information.full_information = full_information;
information.target_information = target_information;
information.conditional_information = conditional_information;
information.full_information_rank = full_rank;
information.target_information_rank = target_rank;
information.raw_generalized_information_eigenvalues = ...
    raw_generalized_eigenvalues;
information.generalized_information_eigenvalues = generalized_eigenvalues;
information.minimum_retained_generalized_eigenvalue = ...
    minimum_retained_generalized_eigenvalue;
information.maximum_absolute_discarded_generalized_eigenvalue = ...
    maximum_absolute_discarded_generalized_eigenvalue;
information.total_target_information_fraction = total_target_information_fraction;
information.visible_direction_information_fraction = ...
    visible_direction_information_fraction;
information.minimum_conditional_information_eigenvalue = ...
    minimum_conditional_eigenvalue;
information.full_information_tolerance = tolerance;
information.full_support_relative_tolerance = ...
    opts.full_support_relative_tolerance;
information.full_support_absolute_tolerance = ...
    opts.full_support_absolute_tolerance;
information.generalized_rank_tolerance = opts.generalized_rank_tolerance;
information.eigenvalue_clip_tolerance = opts.eigenvalue_clip_tolerance;
information.information_tolerance = tolerance; % Backward-compatible alias.
information.precision_audit = audit;
end

function opts = parse_options(varargin)
opts = struct('full_support_relative_tolerance', 1e-8, ...
    'full_support_absolute_tolerance', 1e-12, ...
    'generalized_rank_tolerance', 1e-8, ...
    'eigenvalue_clip_tolerance', 1e-8);
if mod(numel(varargin), 2) ~= 0
    error('Options must be supplied as name/value pairs.');
end
for i = 1:2:numel(varargin)
    name = lower(char(varargin{i}));
    value = varargin{i + 1};
    switch name
        case 'relative_tolerance'
            % Legacy option: preserve its former dual role.
            opts.full_support_relative_tolerance = value;
            opts.generalized_rank_tolerance = value;
        case 'absolute_tolerance'
            opts.full_support_absolute_tolerance = value;
        case 'clip_tolerance'
            opts.eigenvalue_clip_tolerance = value;
        otherwise
            if ~isfield(opts, name), error('Unknown option: %s', name); end
            opts.(name) = value;
    end
end
end

function covariances = normalize_covariances(value, dimension)
if isnumeric(value) && ndims(value) == 3
    covariances = cell(size(value, 3), 1);
    for subject = 1:size(value, 3)
        covariances{subject} = value(:, :, subject);
    end
elseif iscell(value)
    covariances = value(:);
else
    error('subject_covariances must be a cell array or D-by-D-by-N array.');
end
if isempty(covariances), error('At least one subject covariance is required.'); end
for subject = 1:numel(covariances)
    covariance = symmetrize(full(covariances{subject}));
    if ~isequal(size(covariance), [dimension dimension])
        error('Subject covariance %d has the wrong size.', subject);
    end
    if min(real(eig(covariance))) <= 0
        error('Subject covariance %d is not positive definite.', subject);
    end
    covariances{subject} = covariance;
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
