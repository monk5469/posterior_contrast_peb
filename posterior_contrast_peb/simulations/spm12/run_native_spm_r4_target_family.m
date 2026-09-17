function run_native_spm_r4_target_family(varargin)
%RUN_NATIVE_SPM_R4_TARGET_FAMILY All-pattern Native-SPM R=4 target family.
%
% A target model fixes selected *differences* to zero while leaving their
% common modes free.  A standard source-coordinate inclusion model instead
% fixes source coefficients to zero.  These are distinct model families when
% common-mode effects are nonzero.  A rotated full-space encoding is included
% as the matched equivalence control. This script is separate from the prior
% R=4 target-family construction.

script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(script_dir));
addpath(script_dir);
addpath(fullfile(project_dir, 'core'));
if exist('spm_dcm_peb', 'file') ~= 2 || exist('spm_dcm_peb_bmc', 'file') ~= 2
    error('SPM12 spm_dcm_peb.m and spm_dcm_peb_bmc.m are required.');
end
spm_get_defaults('cmdline', true);

cfg = parse(defaults(), varargin{:});
if ~exist(cfg.out_dir, 'dir'); mkdir(cfg.out_dir); end
if ~isempty(cfg.work_dir)
    if ~exist(cfg.work_dir, 'dir'); mkdir(cfg.work_dir); end
    cd(cfg.work_dir);
end
rng(cfg.seed + 100000 * cfg.batch_id, 'twister');
rows = repmat(empty_row(), 0, 1);

for n_targets = cfg.target_dimensions
    patterns = make_patterns(n_targets, cfg.effect_size);
    if cfg.pattern_by_batch
        if cfg.batch_id > numel(patterns)
            error('batch_id must index one of the %d R=4 target patterns.', numel(patterns));
        end
        patterns = patterns(cfg.batch_id);
    end
    for dim = cfg.full_dimensions
        if dim < 2 * n_targets
            error('full dimension must be at least twice the target dimension.');
        end
        L = target_matrix(n_targets, dim);
        T = target_rotation(n_targets, dim);
        check_static_geometry(L, T, cfg.prior_variance);
        Ktarget = binary_model_space(n_targets);
        Krot = false(size(Ktarget, 1), dim);
        Krot(:, 1:n_targets) = Ktarget;
        Ksource = source_pair_model_space(Ktarget, dim, n_targets);

        for p = 1:numel(patterns)
            pattern = patterns(p);
            for r = 1:cfg.reps_per_batch
                rep = (cfg.batch_id - 1) * cfg.reps_per_batch + r;
                [y, C] = simulate_subject_summaries(dim, n_targets, pattern, cfg);
                [source, target, rotated, V] = build_models(y, C, L, T, cfg);
                if min_eig_sym(V) <= 0
                    error('Projected first-level covariance is not positive definite.');
                end
                if min_eig_sym(target.Cp) < -1e-8 || min_eig_sym(rotated.Cp) < -1e-8
                    error('A second-level posterior covariance failed the PSD check.');
                end

                t0 = tic;
                [p_target, ~] = family_posterior(target, Ktarget);
                seconds_target = toc(t0);
                t0 = tic;
                [p_rotated, ~] = family_posterior(rotated, Krot);
                seconds_rotated = toc(t0);
                if cfg.include_source_pair
                    t0 = tic;
                    [p_source, ~] = family_posterior(source, Ksource);
                    seconds_source = toc(t0);
                else
                    p_source = nan(size(p_target));
                    seconds_source = NaN;
                end

                true_idx = find(ismember(Ktarget, pattern.mask, 'rows'), 1, 'first');
                if isempty(true_idx); error('True target pattern absent from target family.'); end
                row = empty_row();
                row.n_targets = n_targets;
                row.full_dim = dim;
                row.pattern = pattern.name;
                row.target_mask = mask_label(pattern.mask);
                row.rep = rep;
                row.n_target_models = size(Ktarget, 1);
                row.target_family_log2_size = n_targets;
                row.source_coordinate_log2_size = dim;
                row.common_fixed_effect = cfg.common_fixed_effect;
                row.direct_true_model_pp = p_target(true_idx);
                row.rotated_true_model_pp = p_rotated(true_idx);
                row.source_pair_matched_pattern_pp = p_source(true_idx);
                row.direct_selected_true = argmax_index(p_target) == true_idx;
                row.rotated_selected_true = argmax_index(p_rotated) == true_idx;
                row.source_pair_selected_matched_pattern = cfg.include_source_pair && argmax_index(p_source) == true_idx;
                row.direct_rotated_max_abs_difference = max(abs(p_target - p_rotated));
                row.direct_selected_model = mask_label(Ktarget(argmax_index(p_target), :));
                row.rotated_selected_model = mask_label(Ktarget(argmax_index(p_rotated), :));
                if cfg.include_source_pair
                    row.source_pair_selected_model = mask_label(Ktarget(argmax_index(p_source), :));
                end
                row.seconds_direct = seconds_target;
                row.seconds_rotated = seconds_rotated;
                row.seconds_source_pair = seconds_source;
                row.rank_L = rank(L);
                row.rotation_orthogonality_error = norm(T * T' - eye(dim), 'fro');
                row.min_eig_target_V = min_eig_sym(V);
                row.min_eig_direct_Cp = min_eig_sym(target.Cp);
                row.min_eig_rotated_Cp = min_eig_sym(rotated.Cp);
                if dim == cfg.full_dimensions(1) && p == 1 && r == 1
                    row.permutation_checked = true;
                    row.permutation_max_abs_difference = permutation_check(y, C, L, cfg, Ktarget, p_target);
                end
                rows(end + 1, 1) = row; %#ok<AGROW>
            end
        end
    end
end

Trows = struct2table(rows, 'AsArray', true);
suffix = sprintf('%s_batch%02dof%02d', cfg.tag, cfg.batch_id, cfg.n_batches);
out_csv = fullfile(cfg.out_dir, sprintf('native_spm_r4_family_rows_%s.csv', suffix));
out_mat = fullfile(cfg.out_dir, sprintf('native_spm_r4_family_rows_%s.mat', suffix));
writetable(Trows, out_csv);
save(out_mat, 'cfg', 'Trows', '-v7');
fprintf('Completed formal Native-SPM R=4 target-family audit %s with %d rows.\n', suffix, height(Trows));
end

function cfg = defaults()
cfg = struct();
cfg.seed = 20260722;
cfg.batch_id = 1;
cfg.n_batches = 12;
cfg.reps_per_batch = 25;
cfg.tag = 'native_r4_target_family';
cfg.target_dimensions = 4;
cfg.full_dimensions = [12 48 96];
cfg.n_subjects = 30;
cfg.obs_var = 0.05;
cfg.posterior_rho = 0.99;
cfg.common_random_variance = 0.20;
cfg.common_fixed_effect = 0.12;
cfg.effect_size = 0.25;
cfg.prior_variance = 1.0;
cfg.out_dir = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'results','native_spm');
cfg.work_dir = '';
cfg.include_source_pair = false;
cfg.pattern_by_batch = false;
end

function cfg = parse(cfg, varargin)
if mod(numel(varargin), 2) ~= 0; error('Use name-value arguments.'); end
for i = 1:2:numel(varargin)
    key = lower(char(varargin{i}));
    if ~isfield(cfg, key); error('Unknown option: %s', key); end
    cfg.(key) = varargin{i + 1};
end
end

function patterns = make_patterns(r, effect_size)
% The R=4 analysis enumerates every binary target pattern.
masks = binary_model_space(r);
patterns = repmat(struct('name', '', 'mask', [], 'delta', []), size(masks, 1), 1);
for i = 1:size(masks, 1)
    patterns(i).mask = masks(i, :);
    patterns(i).delta = effect_size * masks(i, :);
    patterns(i).name = ['M' mask_label(masks(i, :))];
end
end

function L = target_matrix(r, dim)
L = zeros(r, dim);
for j = 1:r
    L(j, 2*j-1:2*j) = [1 -1];
end
end

function T = target_rotation(r, dim)
% Orthogonal block: target differences first, then common modes, then nuisance.
T = eye(dim);
T(1:2*r, 1:2*r) = 0;
for j = 1:r
    T(j, 2*j-1:2*j) = [1 -1] / sqrt(2);
    T(r+j, 2*j-1:2*j) = [1 1] / sqrt(2);
end
end

function K = binary_model_space(r)
K = dec2bin(0:(2^r - 1)) - '0';
end

function Ksource = source_pair_model_space(Ktarget, dim, r)
Ksource = false(size(Ktarget, 1), dim);
for j = 1:r
    Ksource(:, 2*j-1) = Ktarget(:, j);
    Ksource(:, 2*j) = Ktarget(:, j);
end
end

function [y, C] = simulate_subject_summaries(dim, r, pattern, cfg)
C = cfg.obs_var * eye(dim);
for j = 1:r
    pair = 2*j-1:2*j;
    C(pair(1), pair(2)) = cfg.obs_var * cfg.posterior_rho;
    C(pair(2), pair(1)) = C(pair(1), pair(2));
end
R = chol((C + C') / 2, 'lower');
e = randn(cfg.n_subjects, dim) * R';
y = e;
for j = 1:r
    pair = 2*j-1:2*j;
    common = cfg.common_fixed_effect + sqrt(cfg.common_random_variance) * randn(cfg.n_subjects, 1);
    source_mean = [pattern.delta(j) / 2, -pattern.delta(j) / 2];
    y(:, pair) = y(:, pair) + common * [1 1] + repmat(source_mean, cfg.n_subjects, 1);
end
end

function [source, target, rotated, V] = build_models(y, C, L, T, cfg)
dim = size(y, 2);
r = size(L, 1);
if cfg.include_source_pair
    source = spm_dcm_peb(build_gcm(y, C, cfg.prior_variance * eye(dim)), ...
        base_model(size(y, 1), source_target_components(L, dim)), 'B');
else
    source = [];
end

yr = y * T';
Cr = T * C * T';
Pr = T * (cfg.prior_variance * eye(dim)) * T';
rotated = spm_dcm_peb(build_gcm(yr, Cr, Pr), ...
    base_model(size(y, 1), rotated_target_components(r, dim)), 'B');

m = y * L';
V = L * C * L';
opts = struct('parameter_field', 'B', 'prior_variance', ...
    cfg.prior_variance * (L * L'), 'beta', 16, 'noplot', true);
opts.Q = target_only_components(r);
target = spm_dcm_peb_contrast(m, repmat({V}, size(y, 1), 1), ones(size(y, 1), 1), opts);
end

function Q = source_target_components(L, dim)
r = size(L, 1);
Q = cell(r + 1, 1);
Ptarget = zeros(dim);
for j = 1:r
    u = L(j, :)' / norm(L(j, :));
    Q{j} = u * u';
    Ptarget = Ptarget + Q{j};
end
Q{end} = eye(dim) - Ptarget;
end

function Q = rotated_target_components(r, dim)
Q = cell(r + 1, 1);
for j = 1:r
    q = zeros(dim); q(j, j) = 1;
    Q{j} = q;
end
qn = eye(dim); qn(1:r, 1:r) = 0;
Q{end} = qn;
end

function Q = target_only_components(r)
Q = cell(r, 1);
for j = 1:r
    q = zeros(r); q(j, j) = 1;
    Q{j} = q;
end
end

function [p, pip] = family_posterior(PEB, K)
[~, BMR] = spm_dcm_peb_bmc(PEB, K);
F = cellfun(@(x) full(x.F), BMR(:, 1));
p = softmax(F(:))';
pip = K' * p'; %#ok<NASGU>
end

function check_static_geometry(L, T, prior_variance)
r = size(L, 1);
dim = size(L, 2);
if rank(L) ~= r
    error('The R=4 target matrix is not full row rank.');
end
if norm(T * T' - eye(dim), 'fro') > 1e-12
    error('The target/common-mode rotation is not orthogonal.');
end
pC = prior_variance * eye(dim);
if norm(L * pC * L' - 2 * prior_variance * eye(r), 'fro') > 1e-12
    error('The induced direct target prior does not match the declared scale.');
end
rotated_prior = T * pC * T';
if norm(rotated_prior(1:r, 1:r) - prior_variance * eye(r), 'fro') > 1e-12
    error('The rotated target prior does not match the declared scale.');
end
end

function difference = permutation_check(y, C, L, cfg, K, p_reference)
% Reorder target coordinates and masks together; the family posterior must not change.
order = [4 2 1 3];
Lp = L(order, :);
Vp = Lp * C * Lp';
opts = struct('parameter_field', 'B', 'prior_variance', ...
    cfg.prior_variance * (Lp * Lp'), 'beta', 16, 'noplot', true);
opts.Q = target_only_components(4);
PEB = spm_dcm_peb_contrast(y * Lp', repmat({Vp}, size(y, 1), 1), ...
    ones(size(y, 1), 1), opts);
[p_permuted, ~] = family_posterior(PEB, K);
aligned = zeros(size(p_reference));
for i = 1:size(K, 1)
    index = find(ismember(K, K(i, order), 'rows'), 1, 'first');
    aligned(i) = p_permuted(index);
end
difference = max(abs(p_reference - aligned));
if difference > 1e-10
    error('Column-permutation invariance check failed (%.3g).', difference);
end
end

function value = min_eig_sym(A)
value = min(eig((full(A) + full(A)') / 2));
end

function p = softmax(x)
x = x - max(x);
p = exp(x) ./ sum(exp(x));
end

function M = base_model(n, Q)
M = struct('X', ones(n, 1), 'Xnames', {{'mean'}}, ...
    'maxit', 64, 'noplot', true, 'beta', 16);
M.Q = Q;
end

function G = build_gcm(y, C, pC)
[n, d] = size(y);
G = cell(n, 1);
for s = 1:n
    D = struct();
    D.name = sprintf('multitarget_%03d', s);
    D.M.pE.B = zeros(d, 1);
    D.M.pC = pC;
    D.Ep.B = y(s, :)';
    D.Cp = C;
    D.F = 0;
    G{s} = D;
end
end

function idx = argmax_index(x)
[~, idx] = max(x);
end

function label = mask_label(mask)
label = char(mask + '0');
end

function row = empty_row()
row = struct('n_targets', NaN, 'full_dim', NaN, 'pattern', '', ...
    'target_mask', '', 'rep', NaN, 'n_target_models', NaN, ...
    'target_family_log2_size', NaN, 'source_coordinate_log2_size', NaN, ...
    'common_fixed_effect', NaN, 'direct_true_model_pp', NaN, ...
    'rotated_true_model_pp', NaN, 'source_pair_matched_pattern_pp', NaN, ...
    'direct_selected_true', false, 'rotated_selected_true', false, ...
    'source_pair_selected_matched_pattern', false, ...
    'direct_rotated_max_abs_difference', NaN, 'direct_selected_model', '', ...
    'rotated_selected_model', '', 'source_pair_selected_model', '', ...
    'seconds_direct', NaN, 'seconds_rotated', NaN, 'seconds_source_pair', NaN, ...
    'rank_L', NaN, 'rotation_orthogonality_error', NaN, ...
    'min_eig_target_V', NaN, 'min_eig_direct_Cp', NaN, 'min_eig_rotated_Cp', NaN, ...
    'permutation_checked', false, 'permutation_max_abs_difference', NaN);
end
