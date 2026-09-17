function run_native_target_precision_compression_simulation(varargin)
%RUN_NATIVE_TARGET_PRECISION_COMPRESSION_SIMULATION Native-SPM validation.
%
% Compares a target-rank-aware precision dictionary with source-coordinate
% single/all dictionaries, a rotated all-coordinate dictionary, and direct
% target fitting under declared Gaussian generating models.

script_dir = fileparts(mfilename('fullpath'));
spm_dir = locate_spm();
addpath(spm_dir);
addpath(script_dir);
spm('defaults', 'FMRI');
spm_get_defaults('cmdline', true);

cfg = defaults(script_dir);
cfg = parse_options(cfg, varargin{:});
if cfg.batch_id < 1 || cfg.batch_id > cfg.n_batches
    error('batch_id must be between 1 and n_batches.');
end
if exist(cfg.output_dir, 'dir') ~= 7, mkdir(cfg.output_dir); end
if exist(cfg.work_dir, 'dir') ~= 7, mkdir(cfg.work_dir); end
old_dir = pwd;
cleaner = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(cfg.work_dir);
rng(cfg.seed + cfg.batch_id - 1, 'twister');

rows = repmat(empty_row(), 0, 1);
for D = cfg.dims
    for R = cfg.target_dims
        if 2 * R > D, error('D must be at least twice R.'); end
        L = pairwise_contrasts(D, R);
        [rotation, target_map] = target_rotation(L);
        for N = cfg.n_subjects
            for scenario_index = 1:numel(cfg.scenarios)
                scenario = cfg.scenarios{scenario_index};
                for rep_local = 1:cfg.reps_per_batch
                    rep = (cfg.batch_id - 1) * cfg.reps_per_batch + rep_local;
                    [mu, C, pC, true_mean, true_target_cov] = simulate_summaries( ...
                        N, D, R, rotation, target_map, scenario, cfg);
                    F = zeros(N, 1);
                    xnames = {'mean'};
                    X = ones(N, 1);

                    G_source = build_gcm(mu, C, pC, F, 'source');
                    if requested(cfg, 'source_single')
                        [PEB, seconds] = timed_fit(G_source, ...
                            base_model(X, xnames, 'single', cfg), cfg);
                        rows(end + 1, 1) = summarize_fit(PEB, L, 'source_single', ...
                            scenario, D, R, N, rep, seconds, true_mean, ...
                            true_target_cov, cfg, C); %#ok<AGROW>
                    end

                    if requested(cfg, 'source_all')
                        [PEB, seconds] = timed_fit(G_source, ...
                            base_model(X, xnames, 'all', cfg), cfg);
                        rows(end + 1, 1) = summarize_fit(PEB, L, 'source_all', ...
                            scenario, D, R, N, rep, seconds, true_mean, ...
                            true_target_cov, cfg, C); %#ok<AGROW>
                    end

                    [mu_rot, C_rot, pC_rot] = rotate_summaries(mu, C, pC, rotation);
                    G_rot = build_gcm(mu_rot, C_rot, pC_rot, F, 'rotated');
                    projection_rot = [target_map zeros(R, D - R)];

                    if requested(cfg, 'rotated_all')
                        [PEB, seconds] = timed_fit(G_rot, ...
                            base_model(X, xnames, 'all', cfg), cfg);
                        rows(end + 1, 1) = summarize_fit(PEB, projection_rot, ...
                            'rotated_all', scenario, D, R, N, rep, seconds, ...
                            true_mean, true_target_cov, cfg, C_rot); %#ok<AGROW>
                    end

                    if requested(cfg, 'target_rank_compressed')
                        components = targetwise_plus_nuisance_components(D, R);
                        [PEB, seconds] = timed_fit(G_rot, ...
                            base_model(X, xnames, components, cfg), cfg);
                        rows(end + 1, 1) = summarize_fit(PEB, projection_rot, ...
                            'target_rank_compressed', scenario, D, R, N, rep, ...
                            seconds, true_mean, true_target_cov, cfg, C_rot); %#ok<AGROW>
                    end

                    target_mu = mu * L';
                    target_C = cell(N, 1);
                    for subject = 1:N
                        target_C{subject} = symmetrize(L * C{subject} * L');
                    end
                    target_pC = symmetrize(L * pC * L');
                    G_target = build_gcm(target_mu, target_C, target_pC, F, 'target');
                    target_option = 'all';
                    if R == 1, target_option = 'single'; end
                    if requested(cfg, 'target_direct')
                        [PEB, seconds] = timed_fit(G_target, ...
                            base_model(X, xnames, target_option, cfg), cfg);
                        rows(end + 1, 1) = summarize_fit(PEB, eye(R), ...
                            'target_direct', scenario, D, R, N, rep, seconds, ...
                            true_mean, true_target_cov, cfg, target_C); %#ok<AGROW>
                    end
                end
            end
        end
    end
end

T = struct2table(rows);
suffix = sprintf('%s_batch%02dof%02d', cfg.tag, cfg.batch_id, cfg.n_batches);
writetable(T, fullfile(cfg.output_dir, ...
    ['native_target_precision_compression_rows_' suffix '.csv']));
save(fullfile(cfg.output_dir, ...
    ['native_target_precision_compression_rows_' suffix '.mat']), ...
    'cfg', 'T', '-v7');
fprintf('Completed %s with %d rows.\n', suffix, height(T));
end

function cfg = defaults(script_dir)
cfg = struct();
cfg.seed = 20260906;
cfg.batch_id = 1;
cfg.n_batches = 1;
cfg.reps_per_batch = 3;
cfg.tag = 'pilot_v1';
cfg.dims = 12;
cfg.target_dims = [1 2];
cfg.n_subjects = 30;
cfg.scenarios = {'target_sparse', 'global_shared', 'nuisance_heterogeneous'};
cfg.specifications = {'source_single', 'source_all', 'rotated_all', ...
    'target_rank_compressed', 'target_direct'};
cfg.target_tau2 = 0.04;
cfg.target_effect = 0.35;
cfg.obs_var = 0.05;
cfg.obs_log_sd = 0.6;
cfg.obs_correlation = 0.25;
cfg.prior_variance = 1.0;
cfg.beta = 16;
cfg.maxit = 64;
cfg.posterior_draws = 0;
cfg.posterior_seed = 20260907;
cfg.output_dir = script_dir;
cfg.work_dir = fullfile(script_dir, 'work_native_compression_simulation');
end

function cfg = parse_options(cfg, varargin)
if mod(numel(varargin), 2) ~= 0, error('Use name/value options.'); end
for i = 1:2:numel(varargin)
    name = lower(char(varargin{i}));
    if ~isfield(cfg, name), error('Unknown option: %s', name); end
    cfg.(name) = varargin{i + 1};
end
end

function yes = requested(cfg, name)
yes = any(strcmp(cfg.specifications, name));
end

function [mu, C, pC, true_mean, true_target_cov] = simulate_summaries( ...
        N, D, R, rotation, target_map, scenario, cfg)
target_coordinate_cov = diag(cfg.target_tau2 * ones(R, 1));
switch scenario
    case 'target_sparse'
        nuisance_variances = 1e-6 * ones(D - R, 1);
    case 'global_shared'
        nuisance_variances = cfg.target_tau2 * ones(D - R, 1);
    case 'nuisance_heterogeneous'
        nuisance_variances = cfg.target_tau2 * ...
            logspace(log10(0.25), log10(4.0), D - R)';
    otherwise
        error('Unknown scenario: %s', scenario);
end
Psi_rot = blkdiag(target_coordinate_cov, diag(nuisance_variances));
mean_rot = zeros(D, 1);
mean_rot(1:R) = cfg.target_effect ./ diag(target_map);

latent_rot = mvnrnd(mean_rot', Psi_rot, N);
latent_source = latent_rot * rotation;
pC = cfg.prior_variance * eye(D);
C = cell(N, 1);
mu = zeros(N, D);
if abs(cfg.obs_correlation) >= 1
    error('obs_correlation must have absolute value below one.');
end
observation_correlation = toeplitz( ...
    cfg.obs_correlation .^ (0:(D - 1)));
for subject = 1:N
    scale = exp(cfg.obs_log_sd * randn() - 0.5 * cfg.obs_log_sd ^ 2);
    C{subject} = symmetrize(cfg.obs_var * scale * observation_correlation);
    mu(subject, :) = latent_source(subject, :) + ...
        mvnrnd(zeros(1, D), C{subject}, 1);
end
true_mean = L_from_rotation(rotation, target_map) * (rotation' * mean_rot);
true_target_cov = symmetrize(target_map * target_coordinate_cov * target_map');
end

function L = L_from_rotation(rotation, target_map)
R = size(target_map, 1);
L = target_map * rotation(1:R, :);
end

function row = summarize_fit(PEB, projection, specification, scenario, ...
        D, R, N, rep, seconds, true_mean, true_target_cov, cfg, ...
        subject_covariances)
audit = dcm_peb_target_precision_audit(PEB, projection);
information = dcm_peb_target_information_audit( ...
    PEB, projection, subject_covariances);
[estimate, posterior_covariance] = projected_group_posterior(PEB, projection);
joint_posterior_covariance = posterior_covariance;
if isfield(PEB, 'Cp_joint_marginal')
    joint_posterior_covariance = projected_group_posterior_covariance( ...
        PEB.Cp_joint_marginal, projection);
end
target_Ce = symmetrize(projection * full(PEB.Ce) * projection');
errors = estimate - true_mean;
covered = true(R, 1);
joint_covered = true(R, 1);
variance_covered = false(R, 1);
variance_relative_se = nan(R, 1);
sampling_variance_covered = nan(R, 1);
target_covariance_uncertainty = ...
    audit.target_covariance_delta_uncertainty;
for target = 1:R
    covered(target) = abs(errors(target)) <= ...
        1.96 * sqrt(max(posterior_covariance(target, target), 0));
    joint_covered(target) = abs(errors(target)) <= ...
        1.96 * sqrt(max(joint_posterior_covariance(target, target), 0));
    diagonal_vech_index = target * (target + 1) / 2;
    variance_se = sqrt(max(target_covariance_uncertainty( ...
        diagonal_vech_index, diagonal_vech_index), 0));
    variance_covered(target) = abs(target_Ce(target, target) - ...
        true_target_cov(target, target)) <= 1.96 * variance_se;
    variance_relative_se(target) = variance_se / ...
        max(abs(true_target_cov(target, target)), eps);
end
if cfg.posterior_draws >= 100
    posterior = dcm_peb_target_precision_posterior( ...
        PEB, projection, 'n_draws', cfg.posterior_draws, ...
        'seed', cfg.posterior_seed + rep + 1000 * R);
    for target = 1:R
        sampling_variance_covered(target) = ...
            posterior.interval_lower(target, target) <= ...
            true_target_cov(target, target) && ...
            true_target_cov(target, target) <= ...
            posterior.interval_upper(target, target);
    end
end
row = empty_row();
row.scenario = scenario;
row.full_dimension = D;
row.target_dimension = R;
row.n_subjects = N;
row.rep = rep;
row.specification = specification;
row.n_components = audit.n_precision_components;
row.local_target_rank = audit.local_target_rank;
row.local_nullity = audit.local_nullity;
row.null_hyperparameter_uncertainty_fraction = ...
    audit.null_hyperparameter_uncertainty_fraction;
row.total_target_information_fraction = ...
    information.total_target_information_fraction;
row.visible_direction_information_fraction = ...
    information.visible_direction_information_fraction;
row.target_information_rank = information.target_information_rank;
row.propagated_target_covariance_uncertainty_trace = ...
    trace(audit.target_covariance_delta_uncertainty);
row.runtime_seconds = seconds;
row.free_energy = full(PEB.F);
row.mean_squared_error = mean(errors .^ 2);
row.all_targets_covered = all(covered);
row.mean_marginal_coverage = mean(covered);
row.mean_joint_marginal_coverage = mean(joint_covered);
row.joint_to_conditional_trace_ratio = trace(joint_posterior_covariance) / ...
    max(trace(posterior_covariance), eps);
row.hyperparameter_effect_uncertainty_trace = trace( ...
    joint_posterior_covariance - posterior_covariance);
row.mean_target_variance_coverage = mean(variance_covered);
row.mean_target_variance_relative_se = mean(variance_relative_se);
row.mean_target_variance_sampling_coverage = ...
    mean(sampling_variance_covered);
row.target_covariance_relative_error = norm(target_Ce - true_target_cov, 'fro') / ...
    max(norm(true_target_cov, 'fro'), eps);
row.target_mean = encode_matrix(estimate(:)');
row.target_posterior_covariance = encode_matrix(posterior_covariance);
row.target_joint_posterior_covariance = encode_matrix( ...
    joint_posterior_covariance);
row.target_random_effects_covariance = encode_matrix(target_Ce);
row.target_covariance_delta_uncertainty = ...
    encode_matrix(target_covariance_uncertainty);
row.true_target_covariance = encode_matrix(true_target_cov);
row.max_finite_difference_error = audit.max_finite_difference_relative_error;
end

function [estimate, covariance] = projected_group_posterior(PEB, projection)
if isstruct(PEB.Ep), values = full(PEB.Ep.B); else, values = full(PEB.Ep); end
D = size(projection, 2);
values = reshape(values, D, []);
estimate = projection * values(:, 1);
Cp = full(PEB.Cp);
if isvector(Cp), Cp = diag(Cp); end
selector = zeros(size(projection, 1), size(Cp, 1));
selector(:, 1:D) = projection;
covariance = symmetrize(selector * Cp * selector');
end

function covariance = projected_group_posterior_covariance(Cp, projection)
Cp = full(Cp);
if isvector(Cp), Cp = diag(Cp); end
D = size(projection, 2);
selector = zeros(size(projection, 1), size(Cp, 1));
selector(:, 1:D) = projection;
covariance = symmetrize(selector * Cp * selector');
end

function L = pairwise_contrasts(D, R)
L = zeros(R, D);
for target = 1:R
    columns = 2 * target - 1 + (0:1);
    L(target, columns) = [1 -1] / sqrt(2);
end
end

function [rotation, target_map] = target_rotation(L)
[Q, ~] = qr(L', 0);
N = null(Q');
rotation = [Q'; N'];
target_map = L * Q;
if norm(rotation * rotation' - eye(size(rotation, 1)), 'fro') > 1e-10
    error('Rotation is not orthonormal.');
end
end

function [mu_rot, C_rot, pC_rot] = rotate_summaries(mu, C, pC, rotation)
mu_rot = mu * rotation';
C_rot = cell(size(C));
for subject = 1:numel(C)
    C_rot{subject} = symmetrize(rotation * C{subject} * rotation');
end
pC_rot = symmetrize(rotation * pC * rotation');
end

function components = targetwise_plus_nuisance_components(D, R)
components = cell(R + (D > R), 1);
for target = 1:R
    component = zeros(D);
    component(target, target) = 1;
    components{target} = component;
end
if D > R
    nuisance = zeros(D);
    nuisance((R + 1):D, (R + 1):D) = eye(D - R);
    components{R + 1} = nuisance;
end
end

function [PEB, seconds] = timed_fit(GCM, M, cfg)
started = tic;
PEB = spm_dcm_peb(GCM, M, 'B');
seconds = toc(started);
end

function M = base_model(X, xnames, Q, cfg)
M = struct('X', X, 'Xnames', {xnames}, 'maxit', cfg.maxit, ...
    'noplot', true, 'beta', cfg.beta);
M.Q = Q;
end

function GCM = build_gcm(mu, C, pC, F, prefix)
N = size(mu, 1);
D = size(mu, 2);
GCM = cell(N, 1);
for subject = 1:N
    DCM = struct();
    DCM.name = sprintf('%s_%03d', prefix, subject);
    DCM.M.pE = struct('B', zeros(D, 1));
    DCM.M.pC = pC;
    DCM.Ep = struct('B', mu(subject, :)');
    DCM.Cp = C{subject};
    DCM.F = F(subject);
    GCM{subject} = DCM;
end
end

function spm_dir = locate_spm
spm_dir = getenv('SPM12_DIR');
if ~isempty(spm_dir) && exist(fullfile(spm_dir, 'spm.m'), 'file') == 2
    return
end
spm_file = which('spm');
if ~isempty(spm_file)
    spm_dir = fileparts(spm_file);
    return
end
error(['SPM12 could not be located. Add SPM12 to the MATLAB path or ', ...
    'set the SPM12_DIR environment variable.']);
end

function text = encode_matrix(values)
rows = cell(size(values, 1), 1);
for row = 1:size(values, 1)
    rows{row} = strjoin(arrayfun(@(x) sprintf('%.10g', x), ...
        values(row, :), 'UniformOutput', false), ',');
end
text = strjoin(rows, ';');
end

function row = empty_row
row = struct('scenario', '', 'full_dimension', NaN, ...
    'target_dimension', NaN, 'n_subjects', NaN, 'rep', NaN, ...
    'specification', '', 'n_components', NaN, ...
    'local_target_rank', NaN, 'local_nullity', NaN, ...
    'null_hyperparameter_uncertainty_fraction', NaN, ...
    'total_target_information_fraction', NaN, ...
    'visible_direction_information_fraction', NaN, ...
    'target_information_rank', NaN, ...
    'propagated_target_covariance_uncertainty_trace', NaN, ...
    'runtime_seconds', NaN, 'free_energy', NaN, ...
    'mean_squared_error', NaN, 'all_targets_covered', false, ...
    'mean_marginal_coverage', NaN, ...
    'mean_joint_marginal_coverage', NaN, ...
    'joint_to_conditional_trace_ratio', NaN, ...
    'hyperparameter_effect_uncertainty_trace', NaN, ...
    'mean_target_variance_coverage', NaN, ...
    'mean_target_variance_relative_se', NaN, ...
    'mean_target_variance_sampling_coverage', NaN, ...
    'target_covariance_relative_error', NaN, 'target_mean', '', ...
    'target_posterior_covariance', '', ...
    'target_joint_posterior_covariance', '', ...
    'target_random_effects_covariance', '', ...
    'target_covariance_delta_uncertainty', '', ...
    'true_target_covariance', '', 'max_finite_difference_error', NaN);
end

function matrix = symmetrize(matrix)
matrix = (matrix + matrix') / 2;
end
