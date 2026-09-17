function posterior = dcm_peb_target_precision_posterior(PEB, L, varargin)
%DCM_PEB_TARGET_PRECISION_POSTERIOR Propagate PEB precision uncertainty.
%
% Draws second-level log precisions from N(PEB.Eh,PEB.Ch), evaluates
% L*inv(P(g))*L' for each draw, and returns positive, nonlinear posterior
% intervals for the target random-effects covariance. The approximation is
% conditional on the variational Gaussian posterior for the log precisions.

opts = parse_options(varargin{:});
audit = dcm_peb_target_precision_audit(PEB, L);
if ~audit.has_hyperparameter_covariance
    error('PEB.Ch is required for posterior propagation.');
end

Q = PEB.M.Q;
g = full(PEB.Eh(:));
Ch = symmetrize(full(PEB.Ch));
[V, D] = eig(Ch);
eigenvalues = max(real(diag(D)), 0);
factor = real(V * diag(sqrt(eigenvalues)));

previous_rng = rng;
cleanup = onCleanup(@() rng(previous_rng));
rng(opts.seed, 'twister');
g_draws = repmat(g, 1, opts.n_draws) + ...
    factor * randn(numel(g), opts.n_draws);

R = size(L, 1);
n_elements = R * (R + 1) / 2;
covariance_draws = zeros(n_elements, opts.n_draws);
for draw = 1:opts.n_draws
    precision = audit.baseline_precision;
    for k = 1:numel(Q)
        precision = precision + exp(g_draws(k, draw)) * full(Q{k});
    end
    target_covariance = symmetrize(L * stable_inverse(precision) * L');
    covariance_draws(:, draw) = raw_symmetric_vector(target_covariance);
end

mean_vector = mean(covariance_draws, 2);
median_vector = empirical_quantile(covariance_draws, 0.5);
lower_vector = empirical_quantile(covariance_draws, opts.tail_probability / 2);
upper_vector = empirical_quantile(covariance_draws, ...
    1 - opts.tail_probability / 2);

posterior = struct();
posterior.n_draws = opts.n_draws;
posterior.seed = opts.seed;
posterior.tail_probability = opts.tail_probability;
posterior.plugin_target_covariance = audit.target_covariance;
posterior.posterior_mean = raw_symmetric_matrix(mean_vector, R);
posterior.posterior_median = raw_symmetric_matrix(median_vector, R);
posterior.interval_lower = raw_symmetric_matrix(lower_vector, R);
posterior.interval_upper = raw_symmetric_matrix(upper_vector, R);
posterior.delta_standard_error = ...
    audit.target_covariance_delta_standard_error;
posterior.delta_interval_lower = ...
    audit.target_covariance_delta_interval95_lower;
posterior.delta_interval_upper = ...
    audit.target_covariance_delta_interval95_upper;
posterior.audit = audit;
if opts.return_draws
    posterior.covariance_draws = covariance_draws;
else
    posterior.covariance_draws = [];
end
end

function opts = parse_options(varargin)
opts = struct('n_draws', 5000, 'seed', 20260907, ...
    'tail_probability', 0.05, 'return_draws', false);
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
validateattributes(opts.n_draws, {'numeric'}, ...
    {'scalar', 'integer', '>=', 100});
validateattributes(opts.tail_probability, {'numeric'}, ...
    {'scalar', '>', 0, '<', 1});
end

function values = empirical_quantile(draws, probability)
sorted = sort(draws, 2);
n = size(sorted, 2);
position = 1 + (n - 1) * probability;
lower_index = floor(position);
upper_index = ceil(position);
weight = position - lower_index;
values = (1 - weight) * sorted(:, lower_index) + ...
    weight * sorted(:, upper_index);
end

function vector = raw_symmetric_vector(matrix)
n = size(matrix, 1);
vector = zeros(n * (n + 1) / 2, 1);
index = 1;
for column = 1:n
    for row = 1:column
        vector(index) = matrix(row, column);
        index = index + 1;
    end
end
end

function matrix = raw_symmetric_matrix(vector, n)
matrix = zeros(n);
index = 1;
for column = 1:n
    for row = 1:column
        matrix(row, column) = vector(index);
        matrix(column, row) = vector(index);
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
