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
    
    alldata = [];
    for cvi = 1:4% sort the sounds order
        idx_cv = [];
        [~, idx_cv] = sort(allidx(:,cvi));
        alldata(:,:,cvi)= alltmp(idx_cv,:,cvi);
    end
    data2kmean = mean(alldata,3);
    data2kmean_nospeech = data2kmean(49:end,:);
%     data2kmean = zscore(data2kmean);
%     data2kmean_nospeech = zscore(data2kmean_nospeech);
    
    n_cluster = 6;
    
    R = []; W = [];
    [R, W, ~, ~] = nonparametric_ica(data2kmean,...
        n_cluster, 10, 0, 1);
    [R_nospeech, W_nospeech, ~, ~] = nonparametric_ica(data2kmean_nospeech,...
        n_cluster, 10, 0, 1);
     Wperm = [];Rperm = []; Wperm_nospeech = [];Rperm_nospeech = [];
    parfor i = 1:nperms
        Rperm = shuffle(R,1);
        Wperm(:,:,i) = pinv(Rperm)*data2kmean;
        Rperm_nospeech = shuffle(R_nospeech,1);
        Wperm_nospeech(:,:,i) = pinv(Rperm_nospeech)*data2kmean_nospeech;
    end
    
    f1 = figure;f2 = figure;
    %              distributionPlot(squeeze(Wperm(1,1,:)),'histOri','left','color',[0.5 0.5 0.5],'widthDiv',[2 1],...
    %         'addBoxes',0,'showMM',0,'globalNorm',0,'distWidth',0.75)
    for ci = 1:n_cluster
        Wpc = squeeze(Wperm(ci,:,:));
        Wpc = sort(Wpc,2,'descend');
        Wpc_nospeech = squeeze(Wperm_nospeech(ci,:,:));
        Wpc_nospeech = sort(Wpc_nospeech,2,'descend');
        %plot(W(1,:)); hold on;plot(Wpc(:,49),'--')
        idx_sig{ci} = find((W(ci,:)>Wpc(:,floor(p*nperms))'));
        idx_sig_nospeech{ci} = find((W_nospeech(ci,:)>Wpc_nospeech(:,floor(p*nperms))'));

        Wtmp = W(:,idx_sig{ci});
        Wnorm = Wtmp(:,:)./sum(abs(Wtmp),1);
        [Wsort,idx_sort] = sortrows(Wnorm',ci,'descend');
        idx_max = idx_sig{ci}(idx_sort);
        
        Wtmp_nospeech = W_nospeech(:,idx_sig_nospeech{ci});
        Wnorm_nospeech = Wtmp_nospeech(:,:)./sum(abs(Wtmp_nospeech),1);
        [Wsort_nospeech,idx_sort_nospeech] = sortrows(Wnorm_nospeech',ci,'descend');
        idx_max_nospeech = idx_sig_nospeech{ci}(idx_sort_nospeech);
        
        n_total = length(idx_max);
        n_ci(subj,ci) = ceil(prop*n_total);
        n_total_nospeech = length(idx_max_nospeech);
        n_ci_nospeech(subj,ci) = ceil(prop*n_total_nospeech);
        
        dat_test(:,subj,:,ci) = mean(alltest(:,idx_max(1:n_ci(subj,ci)),:),2);% sound * subj * nfold * roi
        dat_train(:,subj,:,ci) = mean(alltrain(:,idx_max(1:n_ci(subj,ci)),:),2);
        dat_test_nospeech(:,subj,:,ci) = mean(alltest_nospeech(:,idx_max_nospeech(1:n_ci_nospeech(subj,ci)),:),2);% sound * subj * nfold * roi
        dat_train_nospeech(:,subj,:,ci) = mean(alltrain_nospeech(:,idx_max_nospeech(1:n_ci_nospeech(subj,ci)),:),2);
        %
        figure(f1)
        subplot(2,3,ci)
        plot(Wsort_nospeech(1:n_ci(subj,ci),:));ylabel('Normalized w');
%         figure(f2)
%         subplot(2,3,ci)
%         bar(R(:,ci));
%         title([' nvoxs: ',num2str(n_total)]);
        pause(0.05)
    end
    sgtitle(f1,['Subj',num2str(subj)]);    
    sgtitle(f2,['Subj',num2str(subj)]);
    thisoutfn1=[savepath,'\figs\SelectVox_subj',num2str(subj),'.tif'];
   % thisoutfn2=[savepath,'\figs\CompProfile_subj',num2str(subj),'.tif'];
    saveas(f1,thisoutfn1);
   % saveas(f2,thisoutfn2);
    close all;
    allW{subj} = W;
    allR{subj} = R;
    vox_sig{subj} = idx_sig;
    Wperms{subj} = Wperm;

    allW_nospeech{subj} = W_nospeech;
    allR_nospeech{subj} = R_nospeech;
    vox_sig_nospeech{subj} = idx_sig_nospeech;
end
cd(savepath)
save(['allroi_Decomp_6comps_',num2str(nperms),'perms.mat'],'n_ci','allR','allR_nospeech',...
    'vox_sig','vox_sig_nospeech','dat_test','dat_train','dat_test','dat_train_nospeech')
%%
PrepareANNReps;