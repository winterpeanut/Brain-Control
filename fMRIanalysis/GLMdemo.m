%% Prepare data
%DataPrepare_fMRI_ANN;
%%
%clear
rootmain='D:\EXP2\AcoSemDNN_Behav_fMRI_Repo\AcoSemDNN_Behav_fMRI_Repo\';
workdir=rootmain;
%cd(workdir)
%run([rootmain,'Install.m'])
addpath([workdir,'code/'])
addpath([workdir,'Toolboxes/'])

outfigdir=[workdir,'LX_results/figures/'];
BLG_mkdir(outfigdir);


modeldist='cos';
dataset='formisano';
%some defaults for the analysis parameters, as specified in the analysis_opt structure
analysis_opt=struct([]);
analysis_opt(1).do_rank=0;
analysis_opt.do_zscore=1;
analysis_opt.do_var_part=0;
analysis_opt.do_permtest=1;
analysis_opt.do_speech=1;

analysis_opt.cv_participants=1;
analysis_opt.n_cvfolds=Inf;
analysis_opt.group_nams=[];
analysis_opt.n_perms=1; 
analysis_opt.save_test_prediction=0;
%some defaults for the plot settings, as specified in the plot_opt structure
plot_opt=struct([]);
plot_opt(1).fun_sound_set=1;%0 = none; 1 = mean
plot_opt.fun_cvs= 2;% 0 = none; 1 = mean; 2 = median
plot_opt.colors_pred=[];
plot_opt.fun_cvs= 0;% 0 = none; 1 = mean; 2 = median
plot_opt.plot_noiseceiling=1;
plot_opt.do_biascorr=0; %correct for permutation bias? default = no
plot_opt.figure_per_roi=1;
plot_opt.roi_names={'HG','PT','PP','mSTG','pSTG','aSTG'};
plot_opt.dataset_name=dataset;
%%%% some examples of analyses; uncomment the relevant one; note that this
%%%% code section starts with a clear statement

%%    analysis_opt.varpart_info=varpart_info;
%%% cross-validate GLMs and do perms
%MTFpred = Pred2(1,:);
maindir = 'D:\EXP2\Results\';
roi_nams = {'HG','PT','PP','mSTG','pSTG','aSTG','allroi'};
demethod = {'Decomp','Kmeans'};
n_clusters = 6;
whichmethod = 1;
zscore = '';% _zscore
whichroi = 7;
dospeech = 1;
oplayer = 'conv4_1';
vggstr = '_randvggish';% _randvggish
if whichmethod == 1
filename = [maindir,'CLUSTERS\',roi_nams{whichroi},'_',demethod{whichmethod},...
           '_',num2str(n_clusters),'comps_1000perms',zscore,'.mat'];
elseif whichmethod == 2
filename = [maindir,'CLUSTERS\',roi_nams{whichroi},'_',demethod{whichmethod},...
           '_',num2str(n_clusters),'comps.mat'];
end
load(filename);
analysis_opt.n_perms=1;
analysis_opt.do_zscore=1;

tic
lambdas = 10.^[1:-0.2:-1];
if dospeech == 1
   analysis_opt.do_speech=1;
   speechstr = '_dospeech';
   Ytest = dat_test;
   Ytrain = dat_train;
   if strcmp(vggstr,'_randvggish')
      X = Pred_rand(strcmp(oplayer,Nams),:);
   else
      X = Pred(strcmp(oplayer,Nams),:); 
   end
else 
 analysis_opt.do_speech=0;
 speechstr = '';
 Ytest = dat_test_nospeech;
 Ytrain = dat_train_nospeech;
   if strcmp(vggstr,'_randvggish')
      X = Pred_nospeech_rand(strcmp(oplayer,Nams),:);
   else
      X = Pred_nospeech(strcmp(oplayer,Nams),:); 
   end
end

out= LX_CV_GLM_fit(X,Nams,Ytrain,Ytest,analysis_opt,lambdas);


% out= LX_CV_GLM_slim(Pred(6,:),dat_train{1},dat_test{1},1);
%cd('D:\EXP2\Results\GLM')
if whichmethod == 1
out_name = [maindir,'GLM\',roi_nams{whichroi},'_',demethod{whichmethod},...
           '_',num2str(n_clusters),'comps_VoxSelect10',zscore,speechstr,vggstr,'_',oplayer,'.mat'];
elseif whichmethod == 2
 out_name = [maindir,'GLM\',roi_nams{whichroi},'_',demethod{whichmethod},vggstr,...
           '_',num2str(n_clusters),'comps_zscore.mat'];   
end
save(out_name,'out','-v7.3');
toc

%%
out2 = out;
rsq = out.results.RSQcv;
nc = out.results.NoiseCeiling;
rsqnew = [];ncnew = [];idx=[];
for sub = 1:5
    for re = 1:6
        for pr = 1
            tmprsq = squeeze(mean(rsq(sub,pr,:,re,1,:),3));
            [subj_comps(sub,re,pr),idx(sub,re,pr)] = max(tmprsq);
            rsqnew(sub,pr,:,re,:) = rsq(sub,pr,:,re,:,idx(sub,re,pr));
            
            tmpnc = squeeze(mean(nc(sub,1,:,re,:),3));
            [ddnc(re,pr),idxnc(re,pr)] = max(tmpnc);
            ncnew(sub,1,:,re,:) = nc(sub,1,:,re,idxnc(re,pr));
        end
    end
end
%outall = out;
out.results.RSQcv = rsqnew;
out.results.NoiseCeiling = ncnew;
%fig_handle=CV_GLM_plot(out.results,plot_opt);
a='done'
%% Plot the cluster profile
savepath = 'D:\EXP2\Results\CLUSTERS';
%subj_comps = squeeze(mean(rsq(:,1,:,:,1,:),3));
pr=1;
xaxis = reshape([1:288],48,[]);
for subj = 1:5
    figure;  
for i = 1:6
    subplot(2,3,i);
    hold on;
    for j = 1:size(xaxis,2)
        if dospeech == 0
           bar(xaxis(:,j),allR_nospeech{subj}(xaxis(:,j),i),'EdgeColor','none'); 
        else
            bar(xaxis(:,j),allR{subj}(xaxis(:,j),i),'EdgeColor','none'); 
        end
        %legend;
    end
    if whichmethod ==1
       nvoxs = length(vox_sig{subj}{i});
    elseif whichmethod ==2
       nvoxs = n_ci(subj,i);
    end
    title(['r^2: ',num2str(subj_comps(subj,i,pr)),' nvoxs: ',num2str(nvoxs)]);
    %title(['nvoxs: ',num2str(n_ci(subj,i))]);
end
  suptitle(['subj',num2str(subj)])
  thisoutfn2=[savepath,'\figs\CompProfileR2_subj',num2str(subj),'_',...
      demethod{whichmethod},zscore,speechstr,vggstr,'.tif'];
 % saveas(gcf,thisoutfn2);
  close all;
end
%% choose the beta

idx_pred = find(ismember(out2.results.PredNams, 'conv4_1'));
idx_roi = find(ismember(plot_opt.roi_names, 'pSTG'));

%rsq: sub * pred * nfold * roi * perms * lambdas 
maxrsq = squeeze(rsq(:,idx_pred,:,idx_roi,50,idx(idx_roi,idx_pred))); 
[maxR,idx_sub] = max(mean(maxrsq,2));
%%
clusters{1} = [2,0,1,3,2];%speech
clusters{2} = [5,5,4,5,4];%music
Betas=[];
idx_sub=4;
idx_roi = [clusters{1}(idx_sub),clusters{2}(idx_sub)];
idx_pred = 1;
%beta: regressors * sub * nfolds * roi * nperms * lambdas
for i = 1:length(idx_roi)
beta = out.results.beta_train{idx_pred};
Beta = mean(squeeze(beta(:,idx_sub,:,idx_roi(i),1,idx(idx_sub,idx_roi(i)))),2);
Betas = cat(2,Betas,Beta);
% h5name = ['D:\python\vggish\','subj',num2str(idx_sub),'_',roi_nams{whichroi},'_',...
%        demethod{whichmethod},'_cluster',num2str(idx_roi),'_6comps_SV.h5'];
% h5create(h5name,'/beta',[4097,1]);
% h5write(h5name,'/beta',Beta);
end
h5name = ['D:\python\vggish\mappingbetas\','subj',num2str(idx_sub),'_',roi_nams{whichroi},'_',...
       demethod{whichmethod},'_6comps_SV10',speechstr,vggstr,'_',oplayer,'.h5'];
h5create(h5name,'/beta',[4097,length(idx_roi)]);
h5write(h5name,'/beta',Betas);


