clearvars -except  HG2hem PP2hem PT2hem mSTG2hem aSTG2hem pSTG2hem
%%
datapath = 'D:\EXP2\fMRIdata';
savepath = 'D:\EXP2\Results\CLUSTERS';
load([datapath,filesep,'Beta2hem.mat'],'HG2hem','PP2hem','PT2hem','mSTG2hem','aSTG2hem','pSTG2hem')
%%
clear dat_train dat_test dat_train_nospeech dat_test_nospeech

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
prop = 0.1;
nperms = 1000;
p = 0.05;
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
    dat_test{subj} = permute(alltest,[1,4,3,2]);
    dat_train{subj} = permute(alltrain,[1,4,3,2]);
    
end
cd(savepath)
%save(['speech_mSTG_voxs',num2str(nperms),'perms.mat'],'dat_test','dat_train','dat_test_nospeech','dat_train_nospeech')
%%
PrepareANNReps;