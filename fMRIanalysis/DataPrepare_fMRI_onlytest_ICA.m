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
nperms = 0;
p = 0.05;
mSTGalldata = [];n_ci=[]; dat_test = []; dat_train=[];dat_test_nospeech = []; dat_train_nospeech=[];
for subj = 1%:length(subj_nams)
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
    testtmp = reshape(permute(alltest,[2,1,3]),size(alltest,2),[]);
    [~,idx_test_tmp] = sort(reshape(idx_test,[],1));
    data2kmean = testtmp(:,idx_test_tmp)';
    data2kmean = zscore(data2kmean);
    alldata = [];
%     for cvi = 1:4% sort the sounds order
%         idx_cv = [];
%         [~, idx_cv] = sort(allidx(:,cvi));
%         alldata(:,:,cvi)= alltmp(idx_cv,:,cvi);
%     end
%     data2kmean = mean(alldata,3);
%     data2kmean_nospeech = data2kmean(49:end,:);
    n_cluster = 6;
    
    R = []; W = [];
    [R, W, ~, ~] = nonparametric_ica(data2kmean,...
        n_cluster, 10, 0, 1);
    
     Wperm = [];Rperm = [];
    parfor i = 1:nperms
        Rperm = shuffle(R,1);
        Wperm(:,:,i) = pinv(Rperm)*data2kmean;
    end
    
    f1 = figure;f2 = figure;
    %              distributionPlot(squeeze(Wperm(1,1,:)),'histOri','left','color',[0.5 0.5 0.5],'widthDiv',[2 1],...
    %         'addBoxes',0,'showMM',0,'globalNorm',0,'distWidth',0.75)
    for ci = 1:n_cluster
        Wpc = squeeze(Wperm(ci,:,:));
        Wpc = sort(Wpc,2,'descend');
        %plot(W(1,:)); hold on;plot(Wpc(:,49),'--')
        idx_sig{ci} = find((W(ci,:)>Wpc(:,floor(p*nperms))'));
        Wtmp = W(:,idx_sig{ci});
        Wnorm = Wtmp(:,:)./sum(abs(Wtmp),1);
        [Wsort,idx_sort] = sortrows(Wnorm',ci,'descend');
        idx_max = idx_sig{ci}(idx_sort);
        
        n_total = length(idx_max);
        n_ci(subj,ci) = ceil(prop*n_total);
        
       dat_test(:,subj,:,ci) = mean(alltest(:,idx_max(1:n_ci(subj,ci)),:),2);% sound * subj * nfold * roi
        dat_train(:,subj,:,ci) = mean(alltrain(:,idx_max(1:n_ci(subj,ci)),:),2);
%         dat_test_nospeech(:,subj,:,ci) = mean(alltest_nospeech(:,idx_max(1:n_ci(subj,ci)),:),2);% sound * subj * nfold * roi
%         dat_train_nospeech(:,subj,:,ci) = mean(alltrain_nospeech(:,idx_max(1:n_ci(subj,ci)),:),2);
        %
        figure(f1)
        subplot(2,3,ci)
        plot(Wsort(1:n_ci(subj,ci),:));ylabel('Normalized w');
        figure(f2)
        subplot(2,3,ci)
        bar(R(:,ci));
        title([' nvoxs: ',num2str(n_total)]);
        pause(0.05)
    end
    sgtitle(f1,['Subj',num2str(subj)]);    
    sgtitle(f2,['Subj',num2str(subj)]);
    thisoutfn1=[savepath,'\figs\SelectVox_subj',num2str(subj),'_dospeech.tif'];
   % thisoutfn2=[savepath,'\figs\CompProfile_subj',num2str(subj),'.tif'];
    saveas(f1,thisoutfn1);
   % saveas(f2,thisoutfn2);
    close all;
    allW{subj} = W;
    allR{subj} = R;
    vox_sig{subj} = idx_sig;
    Wperms{subj} = Wperm;

end
cd(savepath)
%save(['speech_allroi_Decomp_6comps_',num2str(nperms),'perms.mat'],'n_ci','allR','vox_sig','dat_test','dat_train','dat_test_nospeech','dat_train_nospeech')
%%
load('D:\EXP2\AcoSemDNN_Behav_fMRI_Repo\AcoSemDNN_Behav_fMRI_Repo\data\formisano_acoustics\formisano_MTF.mat','MTF_TMabs');
% idx_test(:,1) = HG2hem.CV1.RS1.testSounds;
% idx_test(:,2) = HG2hem.CV2.RS1.testSounds;
% idx_test(:,3) = HG2hem.CV3.RS1.testSounds;
% idx_test(:,4) = HG2hem.CV4.RS1.testSounds;
% idx_train(:,1) = HG2hem.CV1.RS1.trainSounds;
% idx_train(:,2) = HG2hem.CV2.RS1.trainSounds;
% idx_train(:,3) = HG2hem.CV3.RS1.trainSounds;
% idx_train(:,4) = HG2hem.CV4.RS1.trainSounds;

rootmain='D:\EXP2\AcoSemDNN_Behav_fMRI_Repo\AcoSemDNN_Behav_fMRI_Repo\';
workdir = rootmain;
dnn_dir=[workdir,'data\formisano_dnns\vggish\'] ;
d=dir([dnn_dir,'*.hdf5']);
d=struct2cell(d);
fns=d(1,:)';
fns=celfun(@(x) [dnn_dir,x],fns);
layer_nams={'input_3' %1
    'conv1' %2
    'pool1'%3
    'conv2'%4
    'pool2'%5
    'conv3_1'%6
    'conv3_2'%7
    'pool3'%8
    'conv4_1'%9
    'conv4_2'%10
    'pool4'%11
    'flatten'%12
    'fc1_1'%13
    'fc1_2'%14
    'fc2'};%15
%in_layers=[3 5 8 11 13 14 15];
in_layers=[2 4 6 7 9 10 13 14 15];
layer_nams=layer_nams(in_layers);
n_layers=length(layer_nams);
Components=[];
Ndims=[];
Reps = [];
disp(repmat(dataset,[2 5]))
for j=1:length(layer_nams)
    for i=1:length(d)
        tmp=h5read(fns{i},['/',layer_nams{j}]);
        ss=size(tmp);
        if rem(i,5)==0
            str=['sound: ',num2str(i)];
        end
        if ndims(tmp)==3
            %put all non-singleton dimensions first
            tmp = permute(tmp,[3,2,1]);
        elseif ndims(tmp==2)
            tmp = permute(tmp,[2,1]);
        end
        
        if i==1
            dat_tmp=zeros(size(repmat(tmp,[1 1 1 1 length(d)])));
            %             disp(num2str(size(dat_tmp)))
            str=[layer_nams{j},' size: ',num2str(size(dat_tmp))];
            disp(str)
        end
        dat_tmp(:,:,:,:,i)=tmp;
        
    end
    if ss>2
        dat_tmp=mean(dat_tmp,1); %average across analysis frames
    end
    s=size(dat_tmp);
    thisdat_tmp=reshape(dat_tmp,[prod(s(1:4)) s(5)]);
    Reps{j,1} = thisdat_tmp;
    % Reps{1,1} = cat(1,Reps{1,1},thisdat_tmp);
    %         RDMsEuc{end+1,1}=BLG_EucDistND(thisdat_tmp);
    %         RDMsCos{end+1,1}=BLG_CosDistND(thisdat_tmp);
    Components{end+1,1}=[layer_nams{j}];
    Ndims=cat(1,Ndims,size(thisdat_tmp,1));
end

Pred=[];Pred_nospeech=[];
MTF = mean(MTF_TMabs,2);
MTF = reshape(permute(MTF,[5,3,4,1,2]),288,[]);
for cv = 1:4
    Pred{1,1}(:,:,cv) = MTF(idx_train(:,cv),:);
    Pred{1,2}(:,:,cv) = MTF(idx_test(:,cv),:) ;
    for layi = 1:length(Reps)
        Xtrain = Reps{layi}(:,idx_train(:,cv));
        Xtrain_nospeech = Xtrain(:,37:end);
        %Xtrain_nospeech = Reps{layi}(:,idx_train(:,cv)>48);
        Pred{layi+1,1}(:,:,cv) = permute(Xtrain,[2,1]);
        Pred_nospeech{layi+1,1}(:,:,cv) = permute(Xtrain_nospeech,[2,1]);
        
        Xtest = Reps{layi}(:,idx_test(:,cv));
        Xtest_nospeech = Xtest(:,13:end);
        %Xtest_nospeech = Reps{layi}(:,idx_test(:,cv)>48);
        Pred{layi+1,2}(:,:,cv) = permute(Xtest,[2,1]);
        Pred_nospeech{layi+1,2}(:,:,cv) = permute(Xtest_nospeech,[2,1]);
    end
end
Nams{1} = 'MTF';
Nams(2:10) = arrayfun(@(x) layer_nams{x},[1:9],'UniformOutput',false);

clearvars -except Reps HG2hem PP2hem PT2hem mSTG2hem aSTG2hem pSTG2hem Nams Pred dat_test dat_train...
    Pred_nospeech dat_test_nospeech dat_train_nospeech mSTGtrain n_ci allW allR