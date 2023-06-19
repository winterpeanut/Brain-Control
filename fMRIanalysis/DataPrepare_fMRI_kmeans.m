clearvars -except  HG2hem PP2hem PT2hem mSTG2hem aSTG2hem pSTG2hem 
%% 
datapath = 'D:\EXP2\fMRIdata';
load([datapath,filesep,'Beta2hem.mat'],'HG2hem','PP2hem','PT2hem','mSTG2hem','aSTG2hem','pSTG2hem')
%%
clear dat_train dat_test dat_train_nospeech dat_test_nospeech
savepath = 'D:\EXP2\Results\CLUSTERS';

voxs=[];voxsoverlap=[];idx_vox=[];idx_vox_allsubj=[];
subj_nams = {'KV','RS','RS1','RS2','RS3'};
roi_nams = {'HG','PT','PP','mSTG','pSTG','aSTG'};
for subj = 1:length(subj_nams)
    subnam = subj_nams{subj};
    tmp=[];
    for ri = 1:6
        for cvi = 1:4
            r_name = roi_nams{ri};
            voxs_handle = [r_name,'2hem.CV',num2str(cvi),'.',subnam,'.InfoVTC.voxVTC'];
            voxs{cvi,ri} = eval(voxs_handle);
            if cvi==1
                voxsoverlap{ri} = voxs{cvi,ri};
            else
                voxsoverlap{ri} = intersect(voxs{cvi,ri},voxsoverlap{ri});
            end
        end
    end
    tmp = voxsoverlap;
    tmp = repmat(tmp,4,1);
    [~,idx_vox_tmp] = cellfun(@ismember,tmp,voxs,'UniformOutput',false);
    idx_vox_allsubj{subj} = idx_vox_tmp;
end
%%    
method = 'Kmeans';

mSTGalldata = [];n_ci=[]; dat_test = []; dat_train=[];dat_test_nospeech = []; dat_train_nospeech=[];
for subj = 1:length(subj_nams)
    subnam = subj_nams{subj};
    idx_vox = idx_vox_allsubj{subj};
    alltmp = []; alltrain = [];alltest=[];alltrain_nospeech = [];alltest_nospeech=[];
    for ri = 1:6
        r_name = roi_nams{ri};
        mSTGtrain = [];mSTGtest = []; 
        mSTGtrain_nospeech = [];mSTGtest_nospeech = [];
        for cvi = 1:4
            
            if subj == 1
                idx_train_handle = ['mSTG2hem.CV',num2str(cvi),'.',subnam,'.trainSounds;'];
                idx_test_handle = ['mSTG2hem.CV',num2str(cvi),'.',subnam,'.testSounds;'];
                
                idx_train(:,cvi) = eval(idx_train_handle);
                idx_test(:,cvi) = eval(idx_test_handle);
            end
            dat_train_handle = [r_name,'2hem.CV',num2str(cvi),'.',subnam,...
                '.BetasTrain(:,idx_vox{',num2str(cvi),',',num2str(ri),'});'];
            dat_test_handle = [r_name,'2hem.CV',num2str(cvi),'.',subnam,...
                '.BetasTest(:,idx_vox{',num2str(cvi),',',num2str(ri),'});'];
            
            %         dat_train_handle2 = ['mean(',r_name,'2hem.CV',num2str(cvi),'.',subnam,...
            %                             '.BetasTrain(:,idx_vox{',num2str(cvi),',',num2str(ri),'}),2);'];
            %         dat_test_handle2 = ['mean(',r_name,'2hem.CV',num2str(cvi),'.',subnam,...
            %                             '.BetasTest(:,idx_vox{',num2str(cvi),',',num2str(ri),'}),2);'];
            %         dddtrain(:,subj,cvi) = eval(dat_train_handle2) ;
            %         dddtest(:,subj,cvi) = eval(dat_test_handle2) ;
            %
            mSTGtrain(:,:,cvi) = eval(dat_train_handle); % => sound * voxs * subj * fold
            mSTGtest(:,:,cvi) = eval(dat_test_handle);
            
%           mSTGtrain_nospeech(:,:,cvi) = mSTGtrain(37:end,:,cvi);
%           mSTGtest_nospeech(:,:,cvi) = mSTGtest(13:end,:,cvi);
            %mSTGtrain_nospeech(idx_train(:,cvi)<49,:,:)=[];mSTGtest_nospeech(idx_test(:,cvi)<49,:,:)=[];
            
        end
        roitmp = cat(1,mSTGtest,mSTGtrain);
        allidx = cat(1,idx_test,idx_train);
        alltmp = cat(2,alltmp,roitmp);
        alltrain = cat(2,alltrain,mSTGtrain);
        alltest = cat(2,alltest,mSTGtest);
        alltrain_nospeech = alltrain(37:end,:,:);
        alltest_nospeech = alltest(13:end,:,:);
    end
    
    alldata = [];
    for cvi = 1:4% sort the sounds order
        idx_cv = [];
        [~, idx_cv] = sort(allidx(:,cvi));
        alldata(:,:,cvi)= alltmp(idx_cv,:,cvi);
    end
    data2kmean = mean(alldata,3); 
    data2kmean = zscore(data2kmean);
    data2kmean_nospeech = data2kmean(49:end,:); 
    data2kmean_nospeech = zscore(data2kmean_nospeech);
    n_cluster = 6;
      [idx_cluster_nospeech,centroid_nospeech,~,~] = kmeans(data2kmean_nospeech',n_cluster);
      [idx_cluster,centroid,~,~] = kmeans(data2kmean',n_cluster);
              
            
    Cluster = [];
   f1 = figure;%f2 = figure;
    for ci = 1:n_cluster
        n_ci_nospeech(subj,ci)= numel(idx_cluster_nospeech(idx_cluster_nospeech==ci));
        n_ci(subj,ci)= numel(idx_cluster(idx_cluster==ci));

        allR{subj} = centroid'; 
        allR_nospeech{subj} = centroid_nospeech'; 

        figure(f1);
        subplot(2,3,ci);
        bar(centroid_nospeech(ci,:));title(['nvoxs:',num2str(n_ci(subj,ci))]);
%         figure(f2);
%         subplot(2,3,ci);
%         plot(W(ci,:))
        dat_test(:,subj,:,ci) = mean(alltest(:,idx_cluster==ci,:),2);% sound * subj * nfold * roi
        dat_train(:,subj,:,ci) = mean(alltrain(:,idx_cluster==ci,:),2);
        dat_test_nospeech(:,subj,:,ci) = mean(alltest_nospeech(:,idx_cluster_nospeech==ci,:),2);% sound * subj * nfold * roi
        dat_train_nospeech(:,subj,:,ci) = mean(alltrain_nospeech(:,idx_cluster_nospeech==ci,:),2);
    end
    
end
cd(savepath)
save(['allroi_',method,'_6comps.mat'],'n_ci','allR','allR_nospeech','dat_test',...
    'dat_train','dat_test_nospeech','dat_train_nospeech')
%%
PrepareANNReps;