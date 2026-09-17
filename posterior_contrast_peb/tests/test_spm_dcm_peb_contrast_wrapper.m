function test_spm_dcm_peb_contrast_wrapper()
%TEST_SPM_DCM_PEB_CONTRAST_WRAPPER Smoke test for SPM-compatible wrapper.
%
% This test requires SPM on the MATLAB path.

    root_dir = fileparts(fileparts(mfilename('fullpath')));
    addpath(genpath(root_dir));

    if exist('spm_dcm_peb', 'file') ~= 2
        error('test_spm_dcm_peb_contrast_wrapper:MissingSPM', ...
            'SPM function spm_dcm_peb is not on the MATLAB path.');
    end

    spm('defaults', 'fmri');
    spm_get_defaults('cmdline', true);

    m = [0.10; 0.20; 0.15; 0.18; 0.12];
    v = (0.05 * ones(5, 1)).^2;

    options = struct();
    options.true_value = 0.15;
    options.beta = 0;
    options.noplot = true;

    [PEB, GCM, M, diagnostics] = spm_dcm_peb_contrast(m, v, ones(5, 1), options);

    assert(numel(GCM) == 5);
    assert(size(M.X, 1) == 5);
    assert(isfield(PEB, 'Ep'));
    assert(isfield(PEB, 'Cp'));
    assert(abs(full(PEB.Ep(1)) - mean(m)) < 0.02);
    assert(abs(diagnostics.fixed_mean - mean(m)) < 1e-12);

    fprintf('test_spm_dcm_peb_contrast_wrapper passed. PEB Ep %.6f Cp %.6f\n', ...
        full(PEB.Ep(1)), full(PEB.Cp(1, 1)));
end
