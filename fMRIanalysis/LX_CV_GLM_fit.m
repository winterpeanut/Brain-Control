function out=LX_CV_GLM_fit(Preds,Nams,Y_train,Y_test,opt,lambdas)
% function out=fit_CVglms(Preds,Nams,Y_train,Y_test,opt)
%%% function [out,specs]=fit_CVglms(Preds,Y_train,Y_test,specs)
%%% helper function for CV GLM analyses
%%%
%%% Inputs:
%%% Preds    = cell of length {N_GLMs,2}, with predictor groups in rows,
%%%            and train/test stimuli in colums
%%% Nams     = cell structure of GLM names
%%% Ytrain   = dependent variable for training set.
%%%            size: [stimulus_pairs participants stimulus_sets rois]
%%% Ytest    = dependent variable for test set.
%%%            size: [stimulus_pairs participants stimulus_sets rois]
%%% opt      = analysis specs structure with fields (default=0 for all):
%%%               do_rank = binary flag for doing rank GLMs
%%%               do_zscore = flag for zscoring Y and Preds
%%%               do_var_part = 1 for variance partitioning
%%%               cv_participants = 1 for cross-validation across participants
%%%               n_cvfolds       = number of participant CV folds (Inf for all folds)
%%%               n_perms         = number of stimulus permutations per CV fold
%%%                               note: last perm is unpermuted, so
%%%                               n_perm = 1 yields only the plugin estimate
%%%               do_permtest     = 1 if both training and testing Y are
%%%                                 permuted
%%%               cvfolds(optional)= structure with following fields:
%%%                     cvs_test  = participants in test set for each cv fold
%%%                     cvs_train = participants in training set for each cv fold
%%%               input_perms(optional) = structure with following fields:
%%%                     perms_train = stimulus permutation matrix for training set
%%%                     perms_test  = stimulus permutation matrix for test set
%%%  if input_perms is specified, the permutation analyses are carried out using
%%%  the input permutation matrices.
%%%
%%% Output:
%%% out      = output structure with fields:
%%%               opt      = analysis specs structure (see above)
%%%               input_perms    = permutation matrices (see above)
%%%               cvfolds  = partitioning of participants in cv folds (see above)
%%%               results  = analysis results. Structure with fields:
%%%                      PredNams = cell structure of GLM names or variance
%%%                                 partitions (if do_var_part = 1)
%%%                      RSQcv    = RSQcv measures
%%%                      NoiseCeiling = noise ceiling measures
%%%
%%% Bruno L. Giordano
%%% INT, CNRS, Marseille
%%% brungio@gmail.com
%%% - September 2021

opt_defaults={'do_rank' 0
    'do_zscore' 0
    'do_var_part' 0
    'cv_participants' 1
    'n_cvfolds' Inf
    'n_perms' 1
    'do_permtest' 0
    'group_nams' []
    'varpart_info' []
    'cvfolds' []
    'input_perms' []
    'do_speech' 1
    'save_test_prediction' 0
    'save_train_betas' 0
    'do_preds_adjust' 0
    'do_montecarlo_estimate' 0
    'betas_train_input',[]
    'save_adj_terms' 0};

if nargin<5 %specify default opts
    opt=cell2struct(opt_defaults(:,2),opt_defaults(:,1),1);
else
    for i=1:size(opt_defaults,1)
        if ~isfield(opt,opt_defaults{i,1})
            opt=setfield(opt,opt_defaults{i,1},opt_defaults{i,2}); %#ok<SFLD>
        end
    end
end
opt.n_perms=max([1 opt.n_perms]); %at least one permutation = plugin estimate

tmpfields=fieldnames(opt);
for i=1:size(tmpfields,1) %assign to workspace the analysis option variables
    str=[tmpfields{i,1},'=opt.',tmpfields{i,1},';'];
    eval(str)
end
%
% if do_rank %%%check well where to rank, eventually
%     Preds=celfun(@tiedrank,Preds);
% end
%
% if do_zscore
%     Preds=celfun(@zscore,Preds);
%     Y_train=zscore(Y_train);
%     Y_test=zscore(Y_test);
%     %     Y_train=bsxfun(@minus,Y_train,prctile(Y_train,0,1));
%     %     Y_test=bsxfun(@minus,Y_test,prctile(Y_test,0,1));
%     %     Y_train=bsxfun(@times,Y_train,1./prctile(Y_train,100,1));
%     %     Y_test=bsxfun(@times,Y_test,1./prctile(Y_test,100,1));
%     %     Y_train=bsxfun(@minus,Y_train,mean(Y_train,1));
%     %     Y_test=bsxfun(@minus,Y_test,mean(Y_test,1));
% end


[n_stim_test,n_participants,n_stimsets,n_rois]=size(Y_test);
n_models=size(Preds,1);

n_stim_train=size(Y_train,1);

% n_stim_test=ceil(sqrt(n_stimpairs_test*2));
% n_stim_train=ceil(sqrt(n_stimpairs_train*2));


%%% let's prepare the permutations and cv splits
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if n_cvfolds > 0 && isempty(cvfolds) %#ok<NODEF>
%     if isinf(n_cvfolds) %for formisano, all possible cv folds
%         if n_participants~=5
%             error('all possible participant folds only if n_participants = 5 (default = do all folds)')
%         end
%         tmp=nchoosek(1:5,3)';
%         tmp_cvs=zeros(5,size(tmp,2));
%         tmp_cvs(1:3,:)=tmp;
%         for i=1:size(tmp,2)
%             tmp_cvs(4:end,i)=setxor([1:5],tmp_cvs(1:3,i))';
%         end
%     else
%         ncol=1;
%         doindcol=1;
%         dodist=0;
%         tmp_cvs=BLG_Perms(n_participants,ncol,doindcol,dodist,n_cvfolds);
%     end
%     
%     cvfolds=struct([]);
%     n_part_train=ceil(n_participants/2); %ntrain = ntest+1 if n_participants is even
%     %         n_part_test=n_participants-n_part_train;
%     cvfolds(1).cvs_test=tmp_cvs(1:n_part_train,:);
%     cvfolds.cvs_train=tmp_cvs(n_part_train+1:end,:);
% end
% 
% if isinf(n_cvfolds) %for formisano, all possible cv folds
%    n_cvfolds=5;
%    % n_cvfolds=size(cvfolds.cvs_train,2);
%     opt.n_cvfolds=n_cvfolds;
% end
n_cvfolds = n_participants;
%%%%%%%%%%%%%%%%%
if n_perms > 0 && isempty(input_perms)
    if n_stim_test~=72 && n_stim_test~=60 && n_stim_test~=12
        
        ncol=n_stimsets;
        doindcol=1; %different permutation for each condition
        dodist=1; %permute rows and columns of distance matrix
        perms_train=BLG_Perms(n_stim_train,ncol,doindcol,dodist,n_perms);
        perms_train=permute(perms_train,[1 2 4 3]); %[n_stimpairs_train,n_stimsets,1,n_perms]
        if do_permtest
            perms_test=BLG_Perms(n_stim_test,ncol,doindcol,dodist,n_perms);
            perms_test=permute(perms_test,[1 2 4 3]);%[n_stimpairs_train,n_stimsets,1,n_perms]
        else
            perms_test=[];
        end
    else
        %[perms_train,perms_test]=formisano_perms(n_perms,do_speech);
        [perms_train,perms_test]=LX_perms(n_perms,do_speech);
        perms_train=permute(perms_train,[1 2 4 3]); %resort matrix dimensions
        % to follow the format requested by the rest of the code
        perms_test=permute(perms_test,[1 2 4 3]);%resort matrix dimensions
        % to follow the format requested by the rest of the code
        if ~do_permtest
            perms_test=[];
        end
    end
    input_perms(1).n_perms=n_perms;
    input_perms.perms_train=perms_train;
    input_perms.perms_test=perms_test;
end
%%%               input_perms(optional) = structure with following fields:
%%%                     perms_train = stimulus permutation matrix for training set
%%%                     perms_test  = stimulus permutation matrix for test set


% RSQcv=zeros([nsplits,length(Preds),6,nperms_split+1]);
RSQcv=zeros([n_cvfolds,n_models,n_stimsets,n_rois,n_perms,length(lambdas)]);
R2_ols=RSQcv;
% NoiseCeilMain=zeros([nsplits,1,6,nperms_split+1]);
NoiseCeiling=zeros([n_cvfolds,1,n_stimsets,n_rois,length(lambdas)]);
if save_adj_terms
    r2cv=zeros([n_cvfolds,n_models,n_stimsets,n_rois,n_perms]);
    RSQcv_meanslope=zeros([n_cvfolds,n_models,n_stimsets,n_rois,n_perms]);
    RSQcv_adj=zeros([n_cvfolds,n_models,n_stimsets,n_rois,n_perms]);
    n_preds=zeros([1,n_models]);
end
%%

if save_test_prediction==1
    Y_test_pred=zeros([n_stimpairs_test n_cvfolds n_stimsets n_rois n_models]);
    Y_test_obs=Y_test_pred;
elseif save_test_prediction==2
    Y_test_pred=zeros([n_stimpairs_test n_cvfolds n_stimsets n_rois n_models n_perms]);
    Y_test_obs=Y_test_pred;
end


 beta_train=celfun(@(x) zeros([size(x,2)+1 n_cvfolds n_stimsets n_rois n_perms length(lambdas)]),Preds(:,1));


c0=clock;
if cv_participants==1
    for i=1:5
        if rem(i,1)==0
            et=etime(clock,c0);
            disp(['CV folds: ',num2str(i),'/',num2str(n_cvfolds),' etime: ',num2str(et)])
        end
        
        
        %%% prepare Y data for this fold
        fun_main=@(x) permute(mean(x,2),[1 3 4 2]);
        if ~do_zscore && ~do_rank
            fun=fun_main;
        elseif ~do_zscore && do_rank
            fun=@(x) tiedrank(fun_main(x));
        elseif do_zscore && ~do_rank
            fun=@(x) zscore(fun_main(x));
        elseif do_zscore && do_rank
            fun=@(x) zscore(tiedrank(fun_main(x)));
        end
        tmp_Y_train      = fun(Y_train(:,i,:,:));
       
        tmp_Y_test       = fun(Y_test(:,i,:,:));
    
        tmp_Y_train_test = fun(Y_train(:,setxor([1:5],i),:,:)); %for noise ceiling
        tmp_Y_test_train = fun(Y_test(:,setxor([1:5],i),:,:)); %for noise ceiling
       
        
        %         %%% compute noise ceiling: retarded, bonked definition
        %         SSTtest=sum(bsxfun(@minus,tmp_Y_test,mean(tmp_Y_test)).^2);
        %         SSEtest=sum((tmp_Y_test-tmp_Y_test_train).^2,1);
        %         noiseceil=1-SSEtest./SSTtest;
        
        %%% compute noise ceiling, regression generalization definition
        SSTtest=sum(bsxfun(@minus,tmp_Y_test,mean(tmp_Y_test)).^2);
        
        %Predict test-set participants distance using the training-set
        %participants distance while considering training sounds set
        f2=@(x) reshape(permute(x,[1 4 2 3]),[size(x,1) 1 size(x,2)*size(x,3)]);
        
        
        [BetaTrain,~,~]= LX_BLG_GLM_ND(f2(tmp_Y_train),f2(tmp_Y_train_test),0,0,lambdas);
        
        % generalize the prediction to the testing sound set
        fdemean=@(x) bsxfun(@minus,x,mean(x));
        fregrdem=@(x) cat(2,ones(size(x(:,1,:))),fdemean(x)); %demean and add intercept
        Xtest=fregrdem(f2(tmp_Y_test_train));
        
        Pred=mtimesx(Xtest,BetaTrain);
        
        ss=size(tmp_Y_test);
        f3=@(x) reshape(permute(x,[1 3 2 4]),[ss length(lambdas)]);
       
        Pred=f3(Pred);
        SSEtest=sum((tmp_Y_test-Pred).^2,1);
        noiseceil=1-SSEtest./SSTtest;
        NoiseCeiling(i,:,:,:,:)=permute(noiseceil,[1 5 2 3 4]);
      
        
        tmp_Y_train_perms=zeros([n_stim_train,n_stimsets,n_rois,n_perms]);
        for j=1:n_rois
            tmp=tmp_Y_train(:,:,j);
            tmp_Y_train_perms(:,:,j,:)=tmp(input_perms.perms_train);
        end
        
        if do_permtest
            tmp_Y_test_perms=zeros([n_stim_test,n_stimsets,n_rois,n_perms]);
            for j=1:n_rois
                tmp=tmp_Y_test(:,:,j);
                tmp_Y_test_perms(:,:,j,:)=tmp(input_perms.perms_test);
            end
        else
            tmp_Y_test_perms=repmat(tmp_Y_test,[1 1 1 n_perms]);
        end
        
        %%% last perm is plugin
        tmp_Y_test_perms(:,:,:,n_perms)=tmp_Y_test;
        tmp_Y_train_perms(:,:,:,n_perms)=tmp_Y_train;
        
        fun=@(x) permute(x,[1 5 2 3 4]);
        tmp_Y_test_perms=fun(tmp_Y_test_perms);
        tmp_Y_train_perms=fun(tmp_Y_train_perms);
        
        
        SSTtest=sum(bsxfun(@minus,tmp_Y_test_perms,mean(tmp_Y_test_perms)).^2);
        %SSTtrain=sum(bsxfun(@minus,tmp_Y_train_perms,mean(tmp_Y_train_perms)).^2);
        
        for ipred=1:size(Preds,1)
            
            %             if ~isempty(betas_train_input) %#ok<USENS>
            %                 fdemean=@(x) bsxfun(@minus,x,mean(x));
            %                 fregrdem=@(x) cat(2,ones(size(x(:,1,:))),fdemean(x)); %demean and add intercept
            %                 Xtrain=fregrdem(Preds{ipred,1});
            %                 Xtrain=mtimesx(Xtrain,betas_train_input{ipred}); %model is the prediction based on input betas
            %                 [BetaTrain,~,~,~]=BLG_GLM_ND(Xtrain,tmp_Y_train_perms,0,0); %Pred is already the prediction
            %
            %                 Xtest=fregrdem(Preds{ipred,2});
            %                 Xtest=mtimesx(Xtest,betas_train_input{ipred}); %model is the prediction based on input betas
            %                 Xtest=fregrdem(Xtest);
            %                 Pred=mtimesx(Xtest,BetaTrain);
            %             else
            %                 Xtrain=Preds{ipred,1};
            %                 [BetaTrain,~,~,~]=BLG_GLM_ND(Xtrain,tmp_Y_train_perms,0,0); %Pred is already the prediction
            %                 if save_train_betas
            %                     beta_train{ipred}(:,i,:,:,:)=BetaTrain;
            %                 end
            %                 fdemean=@(x) bsxfun(@minus,x,mean(x));
            %                 fregrdem=@(x) cat(2,ones(size(x(:,1,:))),fdemean(x)); %demean and add intercept
            %                 Xtest=fregrdem(Preds{ipred,2});
            %                 Pred=mtimesx(Xtest,BetaTrain);
            %             end
            
            
            %%% directly generalize the input betas
            Xtrain=Preds{ipred,1};
         
            [BetaTrain,~,rsqtmp]=LX_BLG_GLM_inv(Xtrain,tmp_Y_train_perms,0,1,lambdas); %Pred is already the prediction
             beta_train{ipred}(:,i,:,:,:,:)=BetaTrain;
           
            
            fdemean=@(x) bsxfun(@minus,x,mean(x));
            fregrdem=@(x) cat(2,ones(size(x(:,1,:))),fdemean(x)); %demean and add intercept
            Xtest=fregrdem(Preds{ipred,2});
            if ~isempty(betas_train_input) %#ok<USENS>
                %Pred=mtimesx(Xtest,betas_train_input{ipred});
                
                PredTrain=mtimesx(fregrdem(Preds{ipred,1}),betas_train_input{ipred});
                [BetaTransform,~,rsqtmp,~]=BLG_GLM_ND(PredTrain,tmp_Y_train_perms,0,0);
                
                PredTest=mtimesx(Xtest,betas_train_input{ipred});
                sP=size(PredTest);
                PredTest=reshape(PredTest,[sP(1:2) prod(sP(3:end))]);
                PredTest=fregrdem(PredTest);
                PredTest=reshape(PredTest,[sP(1) size(PredTest,2) sP(3:end)]);
                Pred=mtimesx(PredTest,BetaTransform);
                
            else
                Pred=mtimesx(Xtest,BetaTrain);
            end
            
            
            
            if save_test_prediction==1
                %%% Y_test_pred=zeros([n_stimpairs_test n_cvfolds n_stimsets n_rois n_models n_perms]);
                Y_test_pred(:,i,:,:,ipred)=Pred(:,:,:,:,end);
                Y_test_obs(:,i,:,:,ipred)=tmp_Y_test_perms(:,:,:,:,end);
            elseif save_test_prediction==2
                %%% Y_test_pred=zeros([n_stimpairs_test n_cvfolds n_stimsets n_rois n_models n_perms]);
                Y_test_pred(:,i,:,:,ipred,:)=permute(Pred(:,:,:,:,:),[1 2 3 4 6 5]); %end is plugin
                Y_test_obs(:,i,:,:,ipred,:)=permute(tmp_Y_test_perms(:,:,:,:,:),[1 2 3 4 6 5]); %end is plugin
            end
            if isempty(betas_train_input)
                SSEtest=sum((tmp_Y_test_perms-Pred).^2,1);
                tmp=1-SSEtest./SSTtest;
            else
                SSEtest=sum(bsxfun(@minus,tmp_Y_test_perms(:,:,:,:,end),Pred).^2,1);
                tmp=1-SSEtest./SSTtest;
                %                 tmp=BLGmx_corr2(tmp_Y_test_perms(:,:,:,:,end),Pred);
            end
            
            
            %%
            RSQcv(i,ipred,:,:,:,:)=tmp;
            R2_ols(i,ipred,:,:,:,:)=rsqtmp;
            n_preds(ipred)=size(BetaTrain,1)-1;

            if save_adj_terms
                r2cv(i,ipred,:,:,:)=BLGmx_corr2(tmp_Y_test_perms,Pred).^2;
                RSQcv_meanslope(i,ipred,:,:,:)=r2cv(i,ipred,:,:,:)-RSQcv(i,ipred,:,:,:);
                RSQcv_meanslope(i,ipred,:,:,:)=RSQcv_meanslope(i,ipred,:,:,:)./n_preds(ipred);
                RSQcv_adj(i,ipred,:,:,:)=r2cv(i,ipred,:,:,:)-RSQcv_meanslope(i,ipred,:,:,:);
            end
            %%
            
            if true%rem(i,1)==0
                et=etime(clock,c0);
                disp(['---Pred: ',num2str([ipred size(Preds,1)]),' etime: ',num2str(et)])
            end
        end
    end
end


%%% Prepare output
out=struct([]);
out(1).opt=opt;
out.input_perms=input_perms;
out.cvfolds=cvfolds;
out.results=struct([]);
out.results(1).PredNams=Nams;
out.results.NoiseCeiling=NoiseCeiling;
out.results.RSQcv=RSQcv;
out.results.R2_ols=R2_ols;
out.results.n_preds=n_preds;
if save_test_prediction
    out.results.Y_test_pred=Y_test_pred;
    out.results.Y_test_obs=Y_test_obs;
end


out.results.beta_train=beta_train;


if save_adj_terms
    out.results.RSQcv_meanslope=RSQcv_meanslope;
    out.results.r2cv=r2cv;
    out.results.RSQcv_adj=RSQcv_adj;
end

