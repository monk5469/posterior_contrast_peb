function run_native_spm_target_bmr(varargin)
%RUN_NATIVE_SPM_TARGET_BMR_EVIDENCE Native SPM target-hypothesis BMR test.
% Compares equal-prior binary null/target families. The source-pair family is
% included only to demonstrate that it asks a different question.

script_dir = fileparts(mfilename('fullpath'));
project_dir = fileparts(fileparts(script_dir));
addpath(script_dir); addpath(fullfile(project_dir, 'core'));
if exist('spm_dcm_peb_bmc','file') ~= 2, error('SPM12 is required.'); end
cfg = defaults(); cfg = parse(cfg, varargin{:});
if ~isempty(cfg.work_dir), if ~exist(cfg.work_dir,'dir'), mkdir(cfg.work_dir); end, cd(cfg.work_dir); end
rng(cfg.seed + 100000*cfg.batch_id, 'twister');
out_dir = fullfile(project_dir,'results','native_spm'); if exist(out_dir,'dir')~=7, mkdir(out_dir); end
rows = repmat(empty_row(),0,1);
for dim = cfg.dims
    L = zeros(1,dim); L(1:2) = [1 -1]; T = rotation(dim);
    for delta = cfg.deltas
        for si = 1:numel(cfg.scenarios)
            scenario = cfg.scenarios{si}; common = strcmp(scenario,'common_mode') * cfg.common_mean;
            for r = 1:cfg.reps_per_batch
                rep = (cfg.batch_id-1)*cfg.reps_per_batch+r;
                z = sqrt(cfg.tau2)*randn(cfg.n_subjects,dim);
                z(:,1) = z(:,1)+delta; z(:,2) = z(:,2)+common;
                y = z/T' + sqrt(cfg.obs_var)*randn(cfg.n_subjects,dim);
                [full,rot,target] = build_models(y,dim,L,T,cfg);
                [bf_c,pip_c] = binary_bmr(target, logical([0;1]));
                Krot = false(2,dim); Krot(2,1)=true; [bf_r,pip_r] = binary_bmr(rot,Krot);
                Kpair = false(2,dim); Kpair(2,1:2)=true; [bf_p,pip_p] = binary_bmr(full,Kpair);
                rows(end+1,1)=make_row(dim,delta,scenario,rep,'contrast_target_family',bf_c,pip_c); %#ok<AGROW>
                rows(end+1,1)=make_row(dim,delta,scenario,rep,'rotated_target_family',bf_r,pip_r); %#ok<AGROW>
                rows(end+1,1)=make_row(dim,delta,scenario,rep,'source_pair_family',bf_p,pip_p); %#ok<AGROW>
            end
        end
    end
end
Trows=struct2table(rows); suffix=sprintf('%s_batch%02dof%02d',cfg.tag,cfg.batch_id,cfg.n_batches);
save(fullfile(out_dir,sprintf('native_spm_target_bmr_rows_%s.mat',suffix)),'cfg','Trows','-v7');
writetable(Trows,fullfile(out_dir,sprintf('native_spm_target_bmr_rows_%s.csv',suffix)));
fprintf('Completed target-BMR %s with %d rows.\n',suffix,height(Trows));
end

function cfg=defaults()
cfg.seed=20260720; cfg.batch_id=1; cfg.n_batches=20; cfg.reps_per_batch=15; cfg.tag='target_bmr_v1';
cfg.dims=[12 48 192]; cfg.n_subjects=30; cfg.deltas=[0 .35]; cfg.scenarios={'baseline','common_mode'};
cfg.common_mean=.24; cfg.tau2=.04; cfg.obs_var=.05; cfg.prior_variance=1; cfg.work_dir='';
end
function cfg=parse(cfg,varargin)
if mod(numel(varargin),2)~=0,error('Use name-value arguments.');end
for i=1:2:numel(varargin), key=lower(char(varargin{i})); if ~isfield(cfg,key),error('Unknown option %s',key);end, cfg.(key)=varargin{i+1};end
end
function [full,rot,target]=build_models(y,dim,L,T,cfg)
G=build_gcm(y,cfg.obs_var*eye(dim),cfg.prior_variance*eye(dim)); M=base_model(size(y,1),'single'); full=spm_dcm_peb(G,M,'B');
yr=y*T'; C=T*(cfg.obs_var*eye(dim))*T'; P=T*(cfg.prior_variance*eye(dim))*T'; rot=spm_dcm_peb(build_gcm(yr,C,P),M,'B');
m=y*L'; v=repmat(L*(cfg.obs_var*eye(dim))*L',size(y,1),1); o=struct('parameter_field','B','prior_variance',L*(cfg.prior_variance*eye(dim))*L','Q','single','beta',16,'noplot',true);
target=spm_dcm_peb_contrast(m,v,ones(size(y,1),1),o);
end
function [logbf,pip]=binary_bmr(PEB,K)
[~,BMR]=spm_dcm_peb_bmc(PEB,K); logbf=full(BMR{2,1}.F-BMR{1,1}.F); pip=1/(1+exp(-max(min(logbf,700),-700)));
end
function T=rotation(dim),T=eye(dim);T(1,:)=0;T(2,:)=0;T(1,1:2)=[1 -1];T(2,1:2)=[.5 .5];end
function G=build_gcm(y,C,pC)
[n,d]=size(y);G=cell(n,1);for s=1:n,D=struct();D.name=sprintf('bmr_%03d',s);D.M.pE.B=zeros(d,1);D.M.pC=pC;D.Ep.B=y(s,:)';D.Cp=C;D.F=0;G{s}=D;end
end
function M=base_model(n,Q),M=struct('X',ones(n,1),'Xnames',{{'mean'}},'maxit',64,'noplot',true,'beta',16);M.Q=Q;end
function r=empty_row(),r=struct('dim',NaN,'delta',NaN,'scenario','','rep',NaN,'family','','logBF10',NaN,'pip_equal_odds',NaN);end
function r=make_row(dim,delta,scenario,rep,family,bf,pip),r=empty_row();r.dim=dim;r.delta=delta;r.scenario=scenario;r.rep=rep;r.family=family;r.logBF10=bf;r.pip_equal_odds=pip;end
