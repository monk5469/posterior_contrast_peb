function run_release_tests
%RUN_RELEASE_TESTS Execute the public MATLAB numerical contracts.

test_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(test_dir);
addpath(genpath(project_dir));

run(fullfile(test_dir, 'test_dcm_peb_target_precision_audit.m'));
run(fullfile(test_dir, 'test_dcm_peb_target_information_audit.m'));
run(fullfile(test_dir, 'test_dcm_peb_target_precision_posterior.m'));

if exist('spm_dcm_peb', 'file') == 2
    run(fullfile(test_dir, 'test_spm_dcm_peb_contrast_wrapper.m'));
else
    fprintf('SPM12 not on path; skipped SPM-dependent wrapper test.\n');
end

fprintf('All available MATLAB release tests passed.\n');
end
