function [m, V, info] = dcm_apply_posterior_contrast(mu, Sigma, L)
%DCM_APPLY_POSTERIOR_CONTRAST Apply m=L*mu and V=L*Sigma*L'.
%
%   [m, V, INFO] = DCM_APPLY_POSTERIOR_CONTRAST(mu, Sigma, L)
%
%   Supports either a single posterior (numeric mu/Sigma) or a cell array of
%   subject posteriors. For cell input, m and V are returned as cell arrays.

    if nargin < 3
        error('dcm_apply_posterior_contrast:NotEnoughInputs', ...
            'mu, Sigma, and L are required.');
    end

    if iscell(mu)
        if ~iscell(Sigma) || numel(mu) ~= numel(Sigma)
            error('dcm_apply_posterior_contrast:CellSizeMismatch', ...
                'mu and Sigma cell arrays must have the same number of elements.');
        end
        m = cell(size(mu));
        V = cell(size(mu));
        info = cell(size(mu));
        for i = 1:numel(mu)
            [m{i}, V{i}, info{i}] = dcm_apply_posterior_contrast(mu{i}, Sigma{i}, L);
        end
        return;
    end

    mu = mu(:);
    n_params = numel(mu);
    if ~isequal(size(Sigma), [n_params, n_params])
        error('dcm_apply_posterior_contrast:SigmaSizeMismatch', ...
            'Sigma must be %d x %d.', n_params, n_params);
    end
    if size(L, 2) ~= n_params
        error('dcm_apply_posterior_contrast:LSizeMismatch', ...
            'L has %d columns but mu has %d elements.', size(L, 2), n_params);
    end

    m = L * mu;
    V = L * Sigma * L';
    V = (V + V') / 2;

    eig_vals = eig(V);
    info = struct();
    info.n_parameters = n_params;
    info.n_contrasts = size(L, 1);
    info.min_eigenvalue = min(eig_vals);
    info.is_psd_tolerance = info.min_eigenvalue >= -1e-10;
end

