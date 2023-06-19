
datapath = 'D:\EXP2\fMRIdata';
load([datapath,filesep,'Beta2hem.mat'],'HG2hem','PP2hem','PT2hem','mSTG2hem','aSTG2hem','pSTG2hem')
%%
voxs = [];voxs2=[];voxs3=[];voxs4=[];voxsall=[];
idx_vox1=[];idx_vox2=[];idx_vox3=[];idx_vox4=[];
voxs = mSTG2hem.CV1.KV.InfoVTC.voxVTC;
voxs2 = mSTG2hem.CV2.KV.InfoVTC.voxVTC;
voxs3 = mSTG2hem.CV3.KV.InfoVTC.voxVTC;
voxs4 = mSTG2hem.CV4.KV.InfoVTC.voxVTC;
voxsall = intersect(voxs,voxs2);
voxsall = intersect(voxsall,voxs3);
voxsall = intersect(voxsall,voxs4);

idx_c = 2042:2045;
[~,idx_vox1]=ismember(voxsall(idx_c),voxs);
[~,idx_vox2]=ismember(voxsall(idx_c),voxs2);
[~,idx_vox3]=ismember(voxsall(idx_c),voxs3);
[~,idx_vox4]=ismember(voxsall(idx_c),voxs4);


mSTGtrainV(:,1,1,1) = mean(mSTG2hem.CV1.KV.BetasTrain(:,idx_vox1),2);
mSTGtrainV(:,1,2,1) = mean(mSTG2hem.CV2.KV.BetasTrain(:,idx_vox2),2);
mSTGtrainV(:,1,3,1) = mean(mSTG2hem.CV3.KV.BetasTrain(:,idx_vox3),2);
mSTGtrainV(:,1,4,1) = mean(mSTG2hem.CV4.KV.BetasTrain(:,idx_vox4),2);

mSTGtestV(:,1,1,1) = mean(mSTG2hem.CV1.KV.BetasTest(:,idx_vox1),2);
mSTGtestV(:,1,2,1) = mean(mSTG2hem.CV2.KV.BetasTest(:,idx_vox2),2);
mSTGtestV(:,1,3,1) = mean(mSTG2hem.CV3.KV.BetasTest(:,idx_vox3),2);
mSTGtestV(:,1,4,1) = mean(mSTG2hem.CV4.KV.BetasTest(:,idx_vox4),2);

%%
voxs = [];voxs2=[];voxs3=[];voxs4=[];voxsall=[];
idx_vox1=[];idx_vox2=[];idx_vox3=[];idx_vox4=[];

voxs = mSTG2hem.CV1.RS.InfoVTC.voxVTC;
voxs2 = mSTG2hem.CV2.RS.InfoVTC.voxVTC;
voxs3 = mSTG2hem.CV3.RS.InfoVTC.voxVTC;
voxs4 = mSTG2hem.CV4.RS.InfoVTC.voxVTC;
voxsall = intersect(voxs,voxs2);
voxsall = intersect(voxsall,voxs3);
voxsall = intersect(voxsall,voxs4);
idx_c = 2193:2196;
[~,idx_vox1]=ismember(voxsall(idx_c),voxs);
[~,idx_vox2]=ismember(voxsall(idx_c),voxs2);
[~,idx_vox3]=ismember(voxsall(idx_c),voxs3);
[~,idx_vox4]=ismember(voxsall(idx_c),voxs4);

mSTGtrainV(:,2,1,1) = mean(mSTG2hem.CV1.RS.BetasTrain(:,idx_vox1),2);
mSTGtrainV(:,2,2,1) = mean(mSTG2hem.CV2.RS.BetasTrain(:,idx_vox2),2);
mSTGtrainV(:,2,3,1) = mean(mSTG2hem.CV3.RS.BetasTrain(:,idx_vox3),2);
mSTGtrainV(:,2,4,1) = mean(mSTG2hem.CV4.RS.BetasTrain(:,idx_vox4),2);

mSTGtestV(:,2,1,1) = mean(mSTG2hem.CV1.RS.BetasTest(:,idx_vox1),2);
mSTGtestV(:,2,2,1) = mean(mSTG2hem.CV2.RS.BetasTest(:,idx_vox2),2);
mSTGtestV(:,2,3,1) = mean(mSTG2hem.CV3.RS.BetasTest(:,idx_vox3),2);
mSTGtestV(:,2,4,1) = mean(mSTG2hem.CV4.RS.BetasTest(:,idx_vox4),2);
%%
voxs = [];voxs2=[];voxs3=[];voxs4=[];voxsall=[];
idx_vox1=[];idx_vox2=[];idx_vox3=[];idx_vox4=[];

voxs = mSTG2hem.CV1.RS1.InfoVTC.voxVTC;
voxs2 = mSTG2hem.CV2.RS1.InfoVTC.voxVTC;
voxs3 = mSTG2hem.CV3.RS1.InfoVTC.voxVTC;
voxs4 = mSTG2hem.CV4.RS1.InfoVTC.voxVTC;
voxsall = intersect(voxs,voxs2);
voxsall = intersect(voxsall,voxs3);
voxsall = intersect(voxsall,voxs4);

idx_c = 3770:3773;
[~,idx_vox1]=ismember(voxsall(idx_c),voxs);
[~,idx_vox2]=ismember(voxsall(idx_c),voxs2);
[~,idx_vox3]=ismember(voxsall(idx_c),voxs3);
[~,idx_vox4]=ismember(voxsall(idx_c),voxs4);

mSTGtrainV(:,3,1,1) = mean(mSTG2hem.CV1.RS1.BetasTrain(:,idx_vox),2);
mSTGtrainV(:,3,2,1) = mean(mSTG2hem.CV2.RS1.BetasTrain(:,idx_vox2),2);
mSTGtrainV(:,3,3,1) = mean(mSTG2hem.CV3.RS1.BetasTrain(:,idx_vox3),2);
mSTGtrainV(:,3,4,1) = mean(mSTG2hem.CV4.RS1.BetasTrain(:,idx_vox4),2);

mSTGtestV(:,3,1,1) = mean(mSTG2hem.CV1.RS1.BetasTest(:,idx_vox),2);
mSTGtestV(:,3,2,1) = mean(mSTG2hem.CV2.RS1.BetasTest(:,idx_vox2),2);
mSTGtestV(:,3,3,1) = mean(mSTG2hem.CV3.RS1.BetasTest(:,idx_vox3),2);
mSTGtestV(:,3,4,1) = mean(mSTG2hem.CV4.RS1.BetasTest(:,idx_vox4),2);
%%
voxs = [];voxs2=[];voxs3=[];voxs4=[];voxsall=[];
idx_vox1=[];idx_vox2=[];idx_vox3=[];idx_vox4=[];

voxs = mSTG2hem.CV1.RS2.InfoVTC.voxVTC;
voxs2 = mSTG2hem.CV2.RS2.InfoVTC.voxVTC;
voxs3 = mSTG2hem.CV3.RS2.InfoVTC.voxVTC;
voxs4 = mSTG2hem.CV4.RS2.InfoVTC.voxVTC;
voxsall = intersect(voxs,voxs2);
voxsall = intersect(voxsall,voxs3);
voxsall = intersect(voxsall,voxs4);

idx_c = 2729:2732;
[~,idx_vox1]=ismember(voxsall(idx_c),voxs);
[~,idx_vox2]=ismember(voxsall(idx_c),voxs2);
[~,idx_vox3]=ismember(voxsall(idx_c),voxs3);
[~,idx_vox4]=ismember(voxsall(idx_c),voxs4);

mSTGtrainV(:,4,1,1) = mean(mSTG2hem.CV1.RS2.BetasTrain(:,idx_vox),2);
mSTGtrainV(:,4,2,1) = mean(mSTG2hem.CV2.RS2.BetasTrain(:,idx_vox2),2);
mSTGtrainV(:,4,3,1) = mean(mSTG2hem.CV3.RS2.BetasTrain(:,idx_vox3),2);
mSTGtrainV(:,4,4,1) = mean(mSTG2hem.CV4.RS2.BetasTrain(:,idx_vox4),2);

mSTGtestV(:,4,1,1) = mean(mSTG2hem.CV1.RS2.BetasTest(:,idx_vox),2);
mSTGtestV(:,4,2,1) = mean(mSTG2hem.CV2.RS2.BetasTest(:,idx_vox2),2);
mSTGtestV(:,4,3,1) = mean(mSTG2hem.CV3.RS2.BetasTest(:,idx_vox3),2);
mSTGtestV(:,4,4,1) = mean(mSTG2hem.CV4.RS2.BetasTest(:,idx_vox4),2);


%%
voxs = [];voxs2=[];voxs3=[];voxs4=[];voxsall=[];
idx_vox1=[];idx_vox2=[];idx_vox3=[];idx_vox4=[];

voxs = mSTG2hem.CV1.RS3.InfoVTC.voxVTC;
voxs2 = mSTG2hem.CV2.RS3.InfoVTC.voxVTC;
voxs3 = mSTG2hem.CV3.RS3.InfoVTC.voxVTC;
voxs4 = mSTG2hem.CV4.RS3.InfoVTC.voxVTC;
voxsall = intersect(voxs,voxs2);
voxsall = intersect(voxsall,voxs3);
voxsall = intersect(voxsall,voxs4);

idx_c = 1752:1755;
[~,idx_vox1]=ismember(voxsall(idx_c),voxs);
[~,idx_vox2]=ismember(voxsall(idx_c),voxs2);
[~,idx_vox3]=ismember(voxsall(idx_c),voxs3);
[~,idx_vox4]=ismember(voxsall(idx_c),voxs4);

mSTGtrainV(:,5,1,1) = mean(mSTG2hem.CV1.RS3.BetasTrain(:,idx_vox),2);
mSTGtrainV(:,5,2,1) = mean(mSTG2hem.CV2.RS3.BetasTrain(:,idx_vox2),2);
mSTGtrainV(:,5,3,1) = mean(mSTG2hem.CV3.RS3.BetasTrain(:,idx_vox3),2);
mSTGtrainV(:,5,4,1) = mean(mSTG2hem.CV4.RS3.BetasTrain(:,idx_vox4),2);

mSTGtestV(:,5,1,1) = mean(mSTG2hem.CV1.RS3.BetasTest(:,idx_vox),2);
mSTGtestV(:,5,2,1) = mean(mSTG2hem.CV2.RS3.BetasTest(:,idx_vox2),2);
mSTGtestV(:,5,3,1) = mean(mSTG2hem.CV3.RS3.BetasTest(:,idx_vox3),2);
mSTGtestV(:,5,4,1) = mean(mSTG2hem.CV4.RS3.BetasTest(:,idx_vox4),2);
%% anathor

voxs = [];voxs2=[];voxs3=[];voxs4=[];voxsall=[];
idx_vox1=[];idx_vox2=[];idx_vox3=[];idx_vox4=[];
voxs = mSTG2hem.CV1.KV.InfoVTC.voxVTC;
voxs2 = mSTG2hem.CV2.KV.InfoVTC.voxVTC;
voxs3 = mSTG2hem.CV3.KV.InfoVTC.voxVTC;
voxs4 = mSTG2hem.CV4.KV.InfoVTC.voxVTC;
voxsall = intersect(voxs,voxs2);
voxsall = intersect(voxsall,voxs3);
voxsall = intersect(voxsall,voxs4);
idx_c = 4:7;
[~,idx_vox1]=ismember(voxsall(idx_c),voxs);
[~,idx_vox2]=ismember(voxsall(idx_c),voxs2);
[~,idx_vox3]=ismember(voxsall(idx_c),voxs3);
[~,idx_vox4]=ismember(voxsall(idx_c),voxs4);

mSTGtrainV(:,1,1,2) = mean(mSTG2hem.CV1.KV.BetasTrain(:,idx_vox),2);
mSTGtrainV(:,1,2,2) = mean(mSTG2hem.CV2.KV.BetasTrain(:,idx_vox2),2);
mSTGtrainV(:,1,3,2) = mean(mSTG2hem.CV3.KV.BetasTrain(:,idx_vox3),2);
mSTGtrainV(:,1,4,2) = mean(mSTG2hem.CV4.KV.BetasTrain(:,idx_vox4),2);

mSTGtestV(:,1,1,2) = mean(mSTG2hem.CV1.KV.BetasTest(:,idx_vox),2);
mSTGtestV(:,1,2,2) = mean(mSTG2hem.CV2.KV.BetasTest(:,idx_vox2),2);
mSTGtestV(:,1,3,2) = mean(mSTG2hem.CV3.KV.BetasTest(:,idx_vox3),2);
mSTGtestV(:,1,4,2) = mean(mSTG2hem.CV4.KV.BetasTest(:,idx_vox4),2);

%%
voxs = [];voxs2=[];voxs3=[];voxs4=[];voxsall=[];
idx_vox1=[];idx_vox2=[];idx_vox3=[];idx_vox4=[];

voxs = mSTG2hem.CV1.RS.InfoVTC.voxVTC;
voxs2 = mSTG2hem.CV2.RS.InfoVTC.voxVTC;
voxs3 = mSTG2hem.CV3.RS.InfoVTC.voxVTC;
voxs4 = mSTG2hem.CV4.RS.InfoVTC.voxVTC;
voxsall = intersect(voxs,voxs2);
voxsall = intersect(voxsall,voxs3);
voxsall = intersect(voxsall,voxs4);
idx_c = 8:11;
[~,idx_vox1]=ismember(voxsall(idx_c),voxs);
[~,idx_vox2]=ismember(voxsall(idx_c),voxs2);
[~,idx_vox3]=ismember(voxsall(idx_c),voxs3);
[~,idx_vox4]=ismember(voxsall(idx_c),voxs4);

mSTGtrainV(:,2,1,2) = mean(mSTG2hem.CV1.RS.BetasTrain(:,idx_vox),2);
mSTGtrainV(:,2,2,2) = mean(mSTG2hem.CV2.RS.BetasTrain(:,idx_vox2),2);
mSTGtrainV(:,2,3,2) = mean(mSTG2hem.CV3.RS.BetasTrain(:,idx_vox3),2);
mSTGtrainV(:,2,4,2) = mean(mSTG2hem.CV4.RS.BetasTrain(:,idx_vox4),2);

mSTGtestV(:,2,1,2) = mean(mSTG2hem.CV1.RS.BetasTest(:,idx_vox),2);
mSTGtestV(:,2,2,2) = mean(mSTG2hem.CV2.RS.BetasTest(:,idx_vox2),2);
mSTGtestV(:,2,3,2) = mean(mSTG2hem.CV3.RS.BetasTest(:,idx_vox3),2);
mSTGtestV(:,2,4,2) = mean(mSTG2hem.CV4.RS.BetasTest(:,idx_vox4),2);
%%
voxs = [];voxs2=[];voxs3=[];voxs4=[];voxsall=[];
idx_vox1=[];idx_vox2=[];idx_vox3=[];idx_vox4=[];

voxs = mSTG2hem.CV1.RS1.InfoVTC.voxVTC;
voxs2 = mSTG2hem.CV2.RS1.InfoVTC.voxVTC;
voxs3 = mSTG2hem.CV3.RS1.InfoVTC.voxVTC;
voxs4 = mSTG2hem.CV4.RS1.InfoVTC.voxVTC;
voxsall = intersect(voxs,voxs2);
voxsall = intersect(voxsall,voxs3);
voxsall = intersect(voxsall,voxs4);

idx_c = 4:7;
[~,idx_vox1]=ismember(voxsall(idx_c),voxs);
[~,idx_vox2]=ismember(voxsall(idx_c),voxs2);
[~,idx_vox3]=ismember(voxsall(idx_c),voxs3);
[~,idx_vox4]=ismember(voxsall(idx_c),voxs4);

mSTGtrainV(:,3,1,2) = mean(mSTG2hem.CV1.RS1.BetasTrain(:,idx_vox),2);
mSTGtrainV(:,3,2,2) = mean(mSTG2hem.CV2.RS1.BetasTrain(:,idx_vox2),2);
mSTGtrainV(:,3,3,2) = mean(mSTG2hem.CV3.RS1.BetasTrain(:,idx_vox3),2);
mSTGtrainV(:,3,4,2) = mean(mSTG2hem.CV4.RS1.BetasTrain(:,idx_vox4),2);

mSTGtestV(:,3,1,2) = mean(mSTG2hem.CV1.RS1.BetasTest(:,idx_vox),2);
mSTGtestV(:,3,2,2) = mean(mSTG2hem.CV2.RS1.BetasTest(:,idx_vox2),2);
mSTGtestV(:,3,3,2) = mean(mSTG2hem.CV3.RS1.BetasTest(:,idx_vox3),2);
mSTGtestV(:,3,4,2) = mean(mSTG2hem.CV4.RS1.BetasTest(:,idx_vox4),2);
%%
voxs = [];voxs2=[];voxs3=[];voxs4=[];voxsall=[];
idx_vox1=[];idx_vox2=[];idx_vox3=[];idx_vox4=[];

voxs = mSTG2hem.CV1.RS2.InfoVTC.voxVTC;
voxs2 = mSTG2hem.CV2.RS2.InfoVTC.voxVTC;
voxs3 = mSTG2hem.CV3.RS2.InfoVTC.voxVTC;
voxs4 = mSTG2hem.CV4.RS2.InfoVTC.voxVTC;
voxsall = intersect(voxs,voxs2);
voxsall = intersect(voxsall,voxs3);
voxsall = intersect(voxsall,voxs4);

idx_c = 8:11;
[~,idx_vox1]=ismember(voxsall(idx_c),voxs);
[~,idx_vox2]=ismember(voxsall(idx_c),voxs2);
[~,idx_vox3]=ismember(voxsall(idx_c),voxs3);
[~,idx_vox4]=ismember(voxsall(idx_c),voxs4);

mSTGtrainV(:,4,1,2) = mean(mSTG2hem.CV1.RS2.BetasTrain(:,idx_vox),2);
mSTGtrainV(:,4,2,2) = mean(mSTG2hem.CV2.RS2.BetasTrain(:,idx_vox2),2);
mSTGtrainV(:,4,3,2) = mean(mSTG2hem.CV3.RS2.BetasTrain(:,idx_vox3),2);
mSTGtrainV(:,4,4,2) = mean(mSTG2hem.CV4.RS2.BetasTrain(:,idx_vox4),2);

mSTGtestV(:,4,1,2) = mean(mSTG2hem.CV1.RS2.BetasTest(:,idx_vox),2);
mSTGtestV(:,4,2,2) = mean(mSTG2hem.CV2.RS2.BetasTest(:,idx_vox2),2);
mSTGtestV(:,4,3,2) = mean(mSTG2hem.CV3.RS2.BetasTest(:,idx_vox3),2);
mSTGtestV(:,4,4,2) = mean(mSTG2hem.CV4.RS2.BetasTest(:,idx_vox4),2);


%%
voxs = [];voxs2=[];voxs3=[];voxs4=[];voxsall=[];
idx_vox1=[];idx_vox2=[];idx_vox3=[];idx_vox4=[];

voxs = mSTG2hem.CV1.RS3.InfoVTC.voxVTC;
voxs2 = mSTG2hem.CV2.RS3.InfoVTC.voxVTC;
voxs3 = mSTG2hem.CV3.RS3.InfoVTC.voxVTC;
voxs4 = mSTG2hem.CV4.RS3.InfoVTC.voxVTC;
voxsall = intersect(voxs,voxs2);
voxsall = intersect(voxsall,voxs3);
voxsall = intersect(voxsall,voxs4);

idx_c = 10:13;
[~,idx_vox1]=ismember(voxsall(idx_c),voxs);
[~,idx_vox2]=ismember(voxsall(idx_c),voxs2);
[~,idx_vox3]=ismember(voxsall(idx_c),voxs3);
[~,idx_vox4]=ismember(voxsall(idx_c),voxs4);

mSTGtrainV(:,5,1,2) = mean(mSTG2hem.CV1.RS3.BetasTrain(:,idx_vox),2);
mSTGtrainV(:,5,2,2) = mean(mSTG2hem.CV2.RS3.BetasTrain(:,idx_vox2),2);
mSTGtrainV(:,5,3,2) = mean(mSTG2hem.CV3.RS3.BetasTrain(:,idx_vox3),2);
mSTGtrainV(:,5,4,2) = mean(mSTG2hem.CV4.RS3.BetasTrain(:,idx_vox4),2);

mSTGtestV(:,5,1,2) = mean(mSTG2hem.CV1.RS3.BetasTest(:,idx_vox),2);
mSTGtestV(:,5,2,2) = mean(mSTG2hem.CV2.RS3.BetasTest(:,idx_vox2),2);
mSTGtestV(:,5,3,2) = mean(mSTG2hem.CV3.RS3.BetasTest(:,idx_vox3),2);
mSTGtestV(:,5,4,2) = mean(mSTG2hem.CV4.RS3.BetasTest(:,idx_vox4),2);

%%
clear dat_train dat_test
dat_train = mSTGtrainV;
dat_test = mSTGtestV;
%%
load('D:\EXP2\AcoSemDNN_Behav_fMRI_Repo\AcoSemDNN_Behav_fMRI_Repo\data\formisano_acoustics\formisano_MTF.mat','MTF_TMabs');
idx_test(:,1) = HG2hem.CV1.RS1.testSounds;
idx_test(:,2) = HG2hem.CV2.RS1.testSounds;
idx_test(:,3) = HG2hem.CV3.RS1.testSounds;
idx_test(:,4) = HG2hem.CV4.RS1.testSounds;
idx_train(:,1) = HG2hem.CV1.RS1.trainSounds;
idx_train(:,2) = HG2hem.CV2.RS1.trainSounds;
idx_train(:,3) = HG2hem.CV3.RS1.trainSounds;
idx_train(:,4) = HG2hem.CV4.RS1.trainSounds;

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

Pred=[];
MTF = mean(MTF_TMabs,2);
MTF = reshape(permute(MTF,[5,3,4,1,2]),288,[]);
for cv = 1:4
   Pred{1,1}(:,:,cv) = MTF(idx_train(:,cv),:);
   Pred{1,2}(:,:,cv) = MTF(idx_test(:,cv),:) ;
   for layi = 1:length(Reps)
      Xtrain = Reps{layi}(:,idx_train(:,cv));
      Pred{layi+1,1}(:,:,cv) = permute(Xtrain,[2,1]);
      Xtest = Reps{layi}(:,idx_test(:,cv));
      Pred{layi+1,2}(:,:,cv) = permute(Xtest,[2,1]);
   end
end
Nams{1} = 'MTF';
Nams(2:10) = arrayfun(@(x) layer_nams{x},[1:9],'UniformOutput',false);

clearvars -except Reps HG2hem PP2hem PT2hem mSTG2hem aSTG2hem pSTG2hem Nams Pred dat_test dat_train