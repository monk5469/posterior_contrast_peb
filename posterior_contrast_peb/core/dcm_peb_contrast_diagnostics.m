function diagnostics = dcm_peb_contrast_diagnostics(m, V, options)
%DCM_PEB_CONTRAST_DIAGNOSTICS Diagnostic summaries for posterior contrasts.
%
%   DIAGNOSTICS = DCM_PEB_CONTRAST_DIAGNOSTICS(M, V)
%   DIAGNOSTICS = DCM_PEB_CONTRAST_DIAGNOSTICS(M, V, OPTIONS)
%
%   M is N x Q subject-level posterior contrast means.
%   V can be:
%       - N x 1 vector for a single contrast variance;
%       - N x Q matrix of diagonal variances;
%       - N x 1 cell array of Q x Q covariance matrices.
%
%   OPTIONS may include:
%       true_value      1 x Q vector used for recovery diagnostics
%       free_energy    N x 1 vector of first-level free energies
%       prob_threshold scalar posterior probability threshold, default .95
%
%   This function does not run SPM PEB. It summarizes whether the
%   first-level posterior contrasts are well behaved enough to support
%   second-level interpretation.

    if nargin < 3 || isempty(options)
        options = struct();
    end

    m = double(m);
    if isvector(m)
        m = m(:);
    end
    [n, q] = size(m);
    cov_cells = normalize_covariances(V, n, q);

    true_value = get_option(options, 'true_value', nan(1, q));
    true_value = true_value(:)';
    if numel(true_value) == 1 && q > 1
        true_value = repmat(true_value, 1, q);
    end
    if numel(true_value) ~= q
        error('dcm_peb_contrast_diagnostics:TrueValueSize', ...
            'options.true_value must have one value per contrast.');
    end

    prob_threshold = get_option(options, 'prob_threshold', 0.95);
    free_energy = get_option(options, 'free_energy', nan(n, 1));
    free_energy = free_energy(:);
    if numel(free_energy) ~= n
        error('dcm_peb_contrast_diagnostics:FreeEnergySize', ...
            'options.free_energy must have one value per subject.');
    end

    diag_vars = zeros(n, q);
    min_eigs = zeros(n, 1);
    for s = 1:n
        Cs = cov_cells{s};
        diag_vars(s, :) = diag(Cs)';
        min_eigs(s) = min(eig((Cs + Cs') / 2));
    end
    sd = sqrt(diag_vars);
    z = m ./ sd;
    prob_gt_zero = 1 - normcdf(0, m, sd);
    detected = prob_gt_zero > prob_threshold;

    has_true = all(isfinite(true_value));
    errors = nan(n, q);
    covered = false(n, q);
    if has_true
        errors = m - repmat(true_value, n, 1);
        covered = abs(errors) <= 1.96 * sd;
    end

    weights = inverse_variance_weights(diag_vars);
    fixed_mean = sum(weights .* m, 1);
    fixed_var = 1 ./ sum(1 ./ diag_vars, 1);
    loo_shift = max_leave_one_shift(m, diag_vars, fixed_mean);
    pm = paule_mandel_by_contrast(m, diag_vars);

    diagnostics = struct();
    diagnostics.n_subjects = n;
    diagnostics.n_contrasts = q;
    diagnostics.subject_mean = m;
    diagnostics.subject_variance = diag_vars;
    diagnostics.subject_sd = sd;
    diagnostics.prob_gt_zero = prob_gt_zero;
    diagnostics.detected = detected;
    diagnostics.true_value = true_value;
    diagnostics.error = errors;
    diagnostics.covered_95 = covered;
    diagnostics.fixed_mean = fixed_mean;
    diagnostics.fixed_variance = fixed_var;
    diagnostics.paule_mandel_mean = pm.mean;
    diagnostics.paule_mandel_variance = pm.variance;
    diagnostics.paule_mandel_tau2 = pm.tau2;
    diagnostics.mean_subject_variance = mean(diag_vars, 1);
    diagnostics.median_subject_variance = median(diag_vars, 1);
    diagnostics.max_subject_weight = max(weights, [], 1);
    diagnostics.max_fixed_loo_shift = loo_shift;
    diagnostics.min_cov_eigenvalue = min_eigs;
    diagnostics.n_non_psd_covariances = sum(min_eigs < -1e-10);
    diagnostics.free_energy = free_energy;
    diagnostics.mean_free_energy = mean(free_energy, 'omitnan');
    diagnostics.sd_free_energy = std(free_energy, 0, 'omitnan');
    diagnostics.mean_prob_gt_zero = mean(prob_gt_zero, 1);
    diagnostics.detection_rate = mean(detected, 1);
    if has_true
        diagnostics.subject_bias = mean(errors, 1);
        diagnostics.subject_rmse = sqrt(mean(errors.^2, 1));
        diagnostics.coverage_95 = mean(covered, 1);
    else
        diagnostics.subject_bias = nan(1, q);
        diagnostics.subject_rmse = nan(1, q);
        diagnostics.coverage_95 = nan(1, q);
    end
end

function value = get_option(options, name, default_value)
    if isfield(options, name)
        value = options.(name);
    else
        value = default_value;
    end
end

function cov_cells = normalize_covariances(V, n, q)
    if iscell(V)
        if numel(V) ~= n
            error('dcm_peb_contrast_diagnostics:CovarianceCellSize', ...
                'V cell array must have one covariance per subject.');
        end
        cov_cells = V(:);
        for s = 1:n
            if ~isequal(size(cov_cells{s}), [q, q])
                error('dcm_peb_contrast_diagnostics:CovarianceSize', ...
                    'Each covariance must be Q x Q.');
            end
            cov_cells{s} = double((cov_cells{s} + cov_cells{s}') / 2);
        end
        return;
    end

    V = double(V);
    if isvector(V) && q == 1
        V = V(:);
    end
    if ~isequal(size(V), [n, q])
        error('dcm_peb_contrast_diagnostics:VarianceSize', ...
            'V must be N x Q for diagonal variances, or an N-cell covariance array.');
    end
    cov_cells = cell(n, 1);
    for s = 1:n
        cov_cells{s} = diag(V(s, :));
    end
end

function weights = inverse_variance_weights(vars)
    if any(vars(:) <= 0) || any(~isfinite(vars(:)))
        error('dcm_peb_contrast_diagnostics:InvalidVariance', ...
            'All subject variances must be finite and positive.');
    end
    precision = 1 ./ vars;
    weights = precision ./ sum(precision, 1);
end

function shifts = max_leave_one_shift(m, vars, full_mean)
    [n, q] = size(m);
    shifts = zeros(1, q);
    if n < 3
        shifts(:) = NaN;
        return;
    end
    for j = 1:q
        loo = zeros(n, 1);
        for s = 1:n
            keep = true(n, 1);
            keep(s) = false;
            w = 1 ./ vars(keep, j);
            loo(s) = sum(w .* m(keep, j)) / sum(w);
        end
        shifts(j) = max(abs(loo - full_mean(j)));
    end
end

function pm = paule_mandel_by_contrast(m, vars)
    [~, q] = size(m);
    pm.mean = zeros(1, q);
    pm.variance = zeros(1, q);
    pm.tau2 = zeros(1, q);
    for j = 1:q
        r = paule_mandel_single(m(:, j), vars(:, j));
        pm.mean(j) = r.mean;
        pm.variance(j) = r.variance;
        pm.tau2(j) = r.tau2;
    end
end

function result = paule_mandel_single(values, variances)
    values = values(:);
    variances = variances(:);
    n = numel(values);
    df = n - 1;
    q0 = heterogeneity_q(values, variances, 0);
    if q0 <= df
        tau2 = 0;
    else
        lo = 0;
        hi = max(var(values, 0), eps);
        while heterogeneity_q(values, variances, hi) > df
            hi = hi * 2;
        end
        for iter = 1:80
            mid = (lo + hi) / 2;
            if heterogeneity_q(values, variances, mid) > df
                lo = mid;
            else
                hi = mid;
            end
        end
        tau2 = hi;
    end
    w = 1 ./ (variances + tau2);
    result.mean = sum(w .* values) / sum(w);
    result.variance = 1 / sum(w);
    result.tau2 = tau2;
end

function q = heterogeneity_q(values, variances, tau2)
    w = 1 ./ (variances + tau2);
    mu = sum(w .* values) / sum(w);
    q = sum(w .* (values - mu).^2);
end
