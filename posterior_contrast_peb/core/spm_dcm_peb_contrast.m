function [PEB, GCM, M, diagnostics] = spm_dcm_peb_contrast(m, V, X, options)
%SPM_DCM_PEB_CONTRAST Run SPM PEB on posterior contrasts.
%
%   [PEB, GCM, M, DIAGNOSTICS] = SPM_DCM_PEB_CONTRAST(MEAN, COV, X, OPTIONS)
%
%   MEAN is N x Q subject-level posterior contrast means.
%   COV can be an N x Q diagonal variance matrix or an N-cell array of Q x Q
%   covariance matrices.
%   X is the second-level design matrix. If omitted, an intercept-only model
%   is used.
%
%   This wrapper builds minimal DCM summary structures and calls SPM's
%   spm_dcm_peb. It does not alter the original first-level DCM topology.

    if nargin < 4 || isempty(options)
        options = struct();
    end

    m = double(m);
    if isvector(m)
        m = m(:);
    end
    [n, q] = size(m);
    if nargin < 3 || isempty(X)
        X = ones(n, 1);
    end
    if size(X, 1) ~= n
        error('spm_dcm_peb_contrast:DesignSizeMismatch', ...
            'X must have one row per subject.');
    end

    cov_cells = normalize_covariances(V, n, q);
    parameter_field = get_option(options, 'parameter_field', 'B');
    prior_variance = get_option(options, 'prior_variance', 1);

    GCM = cell(n, 1);
    for s = 1:n
        DCM = struct();
        DCM.name = sprintf('posterior_contrast_subject_%03d', s);
        DCM.M = struct();
        DCM.M.pE = struct();
        DCM.M.pE.(parameter_field) = zeros(q, 1);
        DCM.M.pC = prior_variance * speye(q);
        DCM.Ep = struct();
        DCM.Ep.(parameter_field) = m(s, :)';
        DCM.Cp = cov_cells{s};
        DCM.F = get_subject_free_energy(options, s);
        GCM{s, 1} = DCM;
    end

    M = struct();
    M.X = X;
    M.Xnames = get_option(options, 'Xnames', default_xnames(size(X, 2)));
    M.Q = get_option(options, 'Q', 'single');
    M.maxit = get_option(options, 'maxit', 64);
    M.noplot = get_option(options, 'noplot', true);
    if isfield(options, 'second_level_prior_covariance')
        M.pC = options.second_level_prior_covariance;
    elseif isfield(options, 'second_level_prior_variance')
        M.pC = options.second_level_prior_variance * speye(q);
    end
    if isfield(options, 'beta')
        M.beta = options.beta;
    end

    [PEB, ~] = spm_dcm_peb(GCM, M, parameter_field);

    diag_options = struct();
    diag_options.true_value = get_option(options, 'true_value', nan(1, q));
    diag_options.free_energy = get_option(options, 'free_energy', nan(n, 1));
    diagnostics = dcm_peb_contrast_diagnostics(m, cov_cells, diag_options);
end

function value = get_option(options, name, default_value)
    if isfield(options, name)
        value = options.(name);
    else
        value = default_value;
    end
end

function names = default_xnames(p)
    names = cell(1, p);
    for i = 1:p
        names{i} = sprintf('X%d', i);
    end
end

function F = get_subject_free_energy(options, s)
    if isfield(options, 'free_energy') && numel(options.free_energy) >= s
        F = options.free_energy(s);
    else
        F = 0;
    end
end

function cov_cells = normalize_covariances(V, n, q)
    if iscell(V)
        if numel(V) ~= n
            error('spm_dcm_peb_contrast:CovarianceCellSize', ...
                'V cell array must have one covariance per subject.');
        end
        cov_cells = V(:);
        for s = 1:n
            if ~isequal(size(cov_cells{s}), [q, q])
                error('spm_dcm_peb_contrast:CovarianceSize', ...
                    'Each covariance must be Q x Q.');
            end
            cov_cells{s} = full((cov_cells{s} + cov_cells{s}') / 2);
        end
        return;
    end

    V = double(V);
    if isvector(V) && q == 1
        V = V(:);
    end
    if ~isequal(size(V), [n, q])
        error('spm_dcm_peb_contrast:VarianceSize', ...
            'V must be N x Q for diagonal variances, or an N-cell covariance array.');
    end
    cov_cells = cell(n, 1);
    for s = 1:n
        cov_cells{s} = diag(V(s, :));
    end
end
