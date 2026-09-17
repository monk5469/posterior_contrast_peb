function run_native_spm_source_bmr_target_comparison(varargin)
%RUN_NATIVE_SPM_SOURCE_BMR_TARGET_COMPARISON Source-coordinate BMR comparison.
% Tests a weak-source/strong-difference design using Native SPM automatic
% BMR/BMA, a direct target BMR, and a rotated target-coordinate control.
%
% The automatic source BMR is deliberately called without a model matrix:
% [BMA, BMR] = spm_dcm_peb_bmc(PEB). In SPM12 this invokes the greedy
% source-coordinate second-level BMR/BMA search. Its output is compared with
% target-family evidence; it is not described as making parameters disappear.

script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(script_dir));
addpath(script_dir);
addpath(fullfile(project_dir,'core'));
if exist('spm_dcm_peb_bmc','file') ~= 2
    error('SPM12 spm_dcm_peb_bmc.m is required.');
end
spm_get_defaults('cmdline',true);

cfg = parse(defaults(),varargin{:});
if ~isempty(cfg.work_dir)
    if exist(cfg.work_dir,'dir') ~= 7, mkdir(cfg.work_dir); end
    cd(cfg.work_dir);
end
rng(cfg.seed + 100000 * cfg.batch_id,'twister');
rows = repmat(empty_row(),0,1);

for dim = cfg.dims
    L = zeros(1,dim); L(1:2) = [1 -1];
    T = target_rotation(dim);
    for si = 1:numel(cfg.scenarios)
        scenario = cfg.scenarios(si);
        for r = 1:cfg.reps_per_batch
            rep = (cfg.batch_id - 1) * cfg.reps_per_batch + r;
            [y,C] = simulate_subject_summaries(dim,scenario,cfg);
            [source,target,rotated] = build_models(y,C,L,T,cfg);

            [BMA,~] = spm_dcm_peb_bmc(source);
            [source_mean,source_sd,source_p1,source_p2] = source_summary(BMA,L,dim);
            [bf_target,pip_target] = binary_bmr(target,logical([0;1]));
            Krot = false(2,dim); Krot(2,1) = true;
            [bf_rot,pip_rot] = binary_bmr(rotated,Krot);

            row = empty_row();
            row.dim = dim;
            row.scenario = scenario.name;
            row.rep = rep;
            row.delta_true = scenario.delta;
            row.rho_first_level = scenario.rho;
            row.common_tau2 = scenario.common_tau2;
            row.source_bma_delta_mean = source_mean;
            row.source_bma_delta_sd = source_sd;
            row.source1_inclusion = source_p1;
            row.source2_inclusion = source_p2;
            row.min_source_inclusion = min(source_p1,source_p2);
            row.target_logBF10 = bf_target;
            row.target_pip_equal_odds = pip_target;
            row.rotated_logBF10 = bf_rot;
            row.rotated_pip_equal_odds = pip_rot;
            rows(end+1,1) = row; %#ok<AGROW>
        end
    end
end

Trows = struct2table(rows);
suffix = sprintf('%s_batch%02dof%02d',cfg.tag,cfg.batch_id,cfg.n_batches);
out_dir=fullfile(project_dir,'results','native_spm'); if exist(out_dir,'dir')~=7, mkdir(out_dir); end
save(fullfile(out_dir,sprintf('source_bmr_target_comparison_rows_%s.mat',suffix)), ...
    'cfg','Trows','-v7');
writetable(Trows,fullfile(out_dir,sprintf('source_bmr_target_comparison_rows_%s.csv',suffix)));
fprintf('Completed source-BMR target comparison %s with %d rows.\n',suffix,height(Trows));
end

function cfg = defaults()
cfg = struct();
cfg.seed = 20260720;
cfg.batch_id = 1;
cfg.n_batches = 12;
cfg.reps_per_batch = 25;
cfg.tag = 'source_bmr_target_comparison';
cfg.dims = [12 48 96];
cfg.n_subjects = 30;
cfg.obs_var = .05;
cfg.prior_variance = 1;
cfg.work_dir = '';
cfg.scenarios = [ ...
    struct('name','null_independent','delta',0,'rho',0,'common_tau2',0), ...
    struct('name','null_common','delta',0,'rho',.99,'common_tau2',.20), ...
    struct('name','weak_source_strong_difference','delta',.25,'rho',.99,'common_tau2',.20), ...
    struct('name','nonzero_independent','delta',.25,'rho',0,'common_tau2',0)];
end

function cfg = parse(cfg,varargin)
if mod(numel(varargin),2) ~= 0
    error('Use name-value arguments.');
end
for i = 1:2:numel(varargin)
    key = lower(char(varargin{i}));
    if ~isfield(cfg,key), error('Unknown option %s',key); end
    cfg.(key) = varargin{i+1};
end
end

function [y,C] = simulate_subject_summaries(dim,scenario,cfg)
C = cfg.obs_var * eye(dim);
C(1,2) = cfg.obs_var * scenario.rho;
C(2,1) = C(1,2);
R = chol((C + C')/2,'lower');
e = randn(cfg.n_subjects,dim) * R';
common = sqrt(scenario.common_tau2) * randn(cfg.n_subjects,1);
mu = zeros(1,dim);
mu(1:2) = [scenario.delta/2, -scenario.delta/2];
y = e + common * [1 1 zeros(1,dim-2)] + repmat(mu,cfg.n_subjects,1);
end

function [source,target,rotated] = build_models(y,C,L,T,cfg)
dim = size(y,2);
G = build_gcm(y,C,cfg.prior_variance * eye(dim));
source = spm_dcm_peb(G,base_model(size(y,1),source_target_components(L,dim)),'B');

yr = y*T';
Cr = T*C*T';
Pr = T*(cfg.prior_variance * eye(dim))*T';
rotated = spm_dcm_peb(build_gcm(yr,Cr,Pr), ...
    base_model(size(y,1),rotated_target_components(dim)),'B');

m = y * L';
v = repmat(L*C*L',size(y,1),1);
o = struct('parameter_field','B','prior_variance', ...
    L*(cfg.prior_variance * eye(dim))*L','Q','single','beta',16,'noplot',true);
target = spm_dcm_peb_contrast(m,v,ones(size(y,1),1),o);
end

function Q = source_target_components(L,dim)
u = L' / norm(L);
Q = {u*u', eye(dim) - u*u'};
end

function Q = rotated_target_components(dim)
q_target = zeros(dim); q_target(1,1) = 1;
q_nuisance = eye(dim); q_nuisance(1,1) = 0;
Q = {q_target,q_nuisance};
end

function T = target_rotation(dim)
T = eye(dim);
T(1,1:2) = [1 -1] / sqrt(2);
T(2,1:2) = [1 1] / sqrt(2);
end

function [mean_delta,sd_delta,p1,p2] = source_summary(BMA,L,dim)
Ep = spm_vec(BMA.Ep);
Cp = full(BMA.Cp);
if size(Cp,1) < dim
    error('Unexpected automatic BMA covariance dimension.');
end
mean_delta = L * Ep(1:dim);
sd_delta = sqrt(max(L * Cp(1:dim,1:dim) * L',0));
Pp = spm_vec(BMA.Pp);
if numel(Pp) < 2
    error('Unexpected automatic BMA inclusion-probability dimension.');
end
p1 = Pp(1);
p2 = Pp(2);
end

function [logbf,pip] = binary_bmr(PEB,K)
[~,BMR] = spm_dcm_peb_bmc(PEB,K);
logbf = full(BMR{2,1}.F - BMR{1,1}.F);
pip = 1/(1 + exp(-max(min(logbf,700),-700)));
end

function M = base_model(n,Q)
M = struct('X',ones(n,1),'Xnames',{{'mean'}}, ...
    'maxit',64,'noplot',true,'beta',16);
M.Q = Q;
end

function G = build_gcm(y,C,pC)
[n,d] = size(y);
G = cell(n,1);
for s = 1:n
    D = struct();
    D.name = sprintf('source_subject_%03d',s);
    D.M.pE.B = zeros(d,1);
    D.M.pC = pC;
    D.Ep.B = y(s,:)';
    D.Cp = C;
    D.F = 0;
    G{s} = D;
end
end

function r = empty_row()
r = struct('dim',NaN,'scenario','','rep',NaN,'delta_true',NaN, ...
    'rho_first_level',NaN,'common_tau2',NaN, ...
    'source_bma_delta_mean',NaN,'source_bma_delta_sd',NaN, ...
    'source1_inclusion',NaN,'source2_inclusion',NaN, ...
    'min_source_inclusion',NaN,'target_logBF10',NaN, ...
    'target_pip_equal_odds',NaN,'rotated_logBF10',NaN, ...
    'rotated_pip_equal_odds',NaN);
end
