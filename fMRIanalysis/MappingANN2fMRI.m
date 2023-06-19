
%% convert dnn activations to common models format
%% and compute distances.
%% Run this code after having fit the DNN models
%% with analyze_04[a,b,c]*.py code
clearvars -except HG2hem PP2hem PT2hem mSTG2hem aSTG2hem pSTG2hem
rootmain='D:\EXP2\AcoSemDNN_Behav_fMRI_Repo\AcoSemDNN_Behav_fMRI_Repo\';
%run([rootmain,'Install.m'])

datdir=[rootmain,'data\'];

%% prepare ANN reps

% dataset='giordano';
% dataset='formisano';
% whichmodel=1;%kell
% whichmodel=2;%vggish
% whichmodel=3;%yamnet
dataset='formisano';

model_name={'kell' 'vggish' 'yamnet'}';
model_name_nicer={'Kell' 'VGGish' 'Yamnet'};
out_dist_dir=[datdir,dataset,'_dnns\'];
dnn_diRS1=celfun(@(x) [out_dist_dir,x,'\'],model_name);
main_out_fn=celfun(@(x) [out_dist_dir,dataset,'_',x],model_name);

dnn_dir=dnn_diRS1{whichmodel};
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
        in_layers=[ 2 4 6 7 9 10 13 14 15];
        
layer_nams=layer_nams(in_layers);
n_layers=length(layer_nams);
Components=[];
Ndims=[];
disp(repmat(dataset,[2 5]))
for j=1:length(layer_nams)
   for i=1:length(d)
            tmp=h5read(fns{i},['/',layer_nams{j}]);
            ss=size(tmp);
            if rem(i,5)==0
                str=['sound: ',num2str(i)];
                %                     disp(str)
            end
            %         str=[layer_nams{j},' size: ',num2str(size(tmp))];
            %         disp(str)
            if whichmodel==1 %for Kell (one single frame analyzed)
                %put a singletone time dimension first
                s=size(tmp,1,2,3,4);
                idx=find(s>1);
                otherdims=setxor(1:4,idx);
                permdims=[otherdims(1) idx otherdims(2:end)];
                tmp=permute(tmp,permdims);
            elseif whichmodel==2 %put time first
                s=size(tmp,1,2,3,4);
                
                if strcmp(dataset,'formisano')
                    %put all non-singleton dimensions first
                    idx=find(s>1);
                    permdims=[idx setxor(1:4,idx)];
                    tmp=permute(tmp,permdims);
                    
                    idx=(find(s==1,1,'first')); %first singleton is time
                    otherdims=setxor(1:4,idx);
                    permdims=[idx otherdims];
                    tmp=permute(tmp,permdims);
                    
                      if ndims(tmp)>2
                           tmp = squeeze(mean(tmp,1));
                      end
                       
                elseif strcmp(dataset,'giordano')
                    idx=(find(s>1,1,'last')); %last_nonsingleton is time
                    otherdims=setxor(1:4,idx);
                    permdims=[idx otherdims];
                    tmp=permute(tmp,permdims);
                end
                %             str=[layer_nams{j},' size: ',num2str(size(tmp))];
            elseif whichmodel==3
                s=size(tmp,1,2,3,4);
                idx=(find(s>1,1,'last')); %last_nonsingleton is time
                otherdims=setxor(1:4,idx);
                permdims=[idx otherdims];
                tmp=permute(tmp,permdims);
                tmp=tmp(1:end-1,:,:,:); %discard last analysis frame
                %             str=[layer_nams{j},' size: ',num2str(size(tmp))];
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
        Reps{j+1,1} = thisdat_tmp;
        Reps{1,1} = cat(1,Reps{1,1},thisdat_tmp);
%         RDMsEuc{end+1,1}=BLG_EucDistND(thisdat_tmp);
%         RDMsCos{end+1,1}=BLG_CosDistND(thisdat_tmp);
        Components{end+1,1}=[layer_nams{j}];
        Ndims=cat(1,Ndims,size(thisdat_tmp,1));
    end
    
 



%% prepare fMRI data
clearvars -except Reps HG2hem PP2hem PT2hem mSTG2hem aSTG2hem pSTG2hem layer_nams
%%
datapath = 'D:\EXP2\fMRIdata'
load([datapath,filesep,'Beta2hem.mat'],'HG2hem','PP2hem','PT2hem','mSTG2hem','aSTG2hem','pSTG2hem')
%%

%%
n_test = size(HGtest,1);
n_folds = 4;
n_rois = 6;
%AllROIs_test = zeros(n_test,n_folds,n_rois);
AllROIs_test = cat(3,PPtest,HGtest,PTtest,pSTGtest,mSTGtest,aSTGtest);
AllROIs_train = cat(3,PPtrain,HGtrain,PTtrain,pSTGtrain,mSTGtrain,aSTGtrain);

%%
idx_test(:,1) = HG2hem.CV1.RS1.testSounds;
idx_test(:,2) = HG2hem.CV2.RS1.testSounds;
idx_test(:,3) = HG2hem.CV3.RS1.testSounds;
idx_test(:,4) = HG2hem.CV4.RS1.testSounds;
idx_train(:,1) = HG2hem.CV1.RS1.trainSounds;
idx_train(:,2) = HG2hem.CV2.RS1.trainSounds;
idx_train(:,3) = HG2hem.CV3.RS1.trainSounds;
idx_train(:,4) = HG2hem.CV4.RS1.trainSounds;

AllLayer_test=[];AllLayer_train=[];
for i = 1:n_folds
AllLayer_test(:,i,:) = Reps{2}(:,idx_test(:,i));
AllLayer_train(:,i,:) = Reps{2}(:,idx_train(:,i));
end
AllLayer_test2=[];AllLayer_train2=[];AllROIs_test2=[];AllROIs_train2=[];
AllLayer_test2 = permute(AllLayer_test,[3,1,2]);
AllLayer_train2 = permute(AllLayer_train,[3,1,2]);
%%
AllROIs_test2 = permute(AllROIs_test,[1,4,2,3]);
AllROIs_train2 = permute(AllROIs_train,[1,4,2,3]);


[Beta,~,RS1Q]=BLG_GLM_ND(AllLayer_train2,AllROIs_train2,0,0);
lambdas = 10.^[0:-1:-49];
 beta = LX_RidgeRegress(AllLayer_train2,AllROIs_train2,lambdas,4);
       
SSTtest=mean(bsxfun(@minus,AllROIs_test2,mean(AllROIs_test2)).^2);

fdemean=@(x) bsxfun(@minus,x,mean(x));
fregrdem=@(x) cat(2,ones(size(x(:,1,:))),fdemean(x)); %demean and add intercept
%Xtest=fregrdem(AllLayer_test2);
Xtest=fdemean(AllLayer_test2);
beta = permute(beta,[1,5,2,3,4]);
Pred=mtimesx(Xtest,beta);
SSEtest=mean(bsxfun(@minus,AllROIs_test2,Pred).^2,1);
findlambda =squeeze(mean(mean(SSEtest,3),4));
[~,idxL] = min(findlambda);

w = beta(:,:,:,:,idxL);
SSEtestL = SSEtest(:,:,:,:,idxL);
tmp=1-SSEtestL./SSTtest; 
Rsq = squeeze(tmp)
mean(Rsq,1)

%%
[Beta,~,RS1Q]=BLG_GLM_ND(AllLayer_train2,AllROIs_train2,0,0);
   
SSTtest=mean(bsxfun(@minus,AllROIs_test2,mean(AllROIs_test2)).^2);

fdemean=@(x) bsxfun(@minus,x,mean(x));
fregrdem=@(x) cat(2,ones(size(x(:,1,:))),fdemean(x)); %demean and add intercept
Xtest=fregrdem(AllLayer_test2);
Pred=mtimesx(Xtest,Beta);
SSEtest=mean(bsxfun(@minus,AllROIs_test2,Pred).^2,1);

tmp=1-SSEtest./SSTtest; 
Rsq = squeeze(tmp)
mean(Rsq,1)

%%
idx_test(:,1) = HG2hem.CV1.RS1.testSounds;
idx_test(:,2) = HG2hem.CV2.RS1.testSounds;
idx_test(:,3) = HG2hem.CV3.RS1.testSounds;
idx_test(:,4) = HG2hem.CV4.RS1.testSounds;
idx_train(:,1) = HG2hem.CV1.RS1.trainSounds;
idx_train(:,2) = HG2hem.CV2.RS1.trainSounds;
idx_train(:,3) = HG2hem.CV3.RS1.trainSounds;
idx_train(:,4) = HG2hem.CV4.RS1.trainSounds;


for layi = 2:length(Reps)
for cv = 1:4
    
Ytrain = pSTGtrain{cv};
Ytest = pSTGtest{cv};
Ytrain = permute(Ytrain,[1,3,2]);
Ytest = permute(Ytest,[1,3,2]);

Xtrain = Reps{layi}(:,idx_train(:,cv));
Xtrain = permute(Xtrain,[2,1]);
Xtest = Reps{layi}(:,idx_test(:,cv));
Xtest = permute(Xtest,[2,1]);

[Beta,~,RS1Q]=BLG_GLM_ND(Xtrain,Ytrain,0,0);
SSTtest=mean(bsxfun(@minus,Ytest,mean(Ytest)).^2);

fdemean=@(x) bsxfun(@minus,x,mean(x));
fregrdem=@(x) cat(2,ones(size(x(:,1,:))),fdemean(x)); %demean and add intercept
Xtest=fregrdem(Xtest);
Pred=mtimesx(Xtest,Beta);
SSEtest=mean(bsxfun(@minus,Ytest,Pred).^2,1);

tmp=1-SSEtest./SSTtest; 
Rsq(layi,cv) = mean(squeeze(tmp));
frsq=@(y,ypred) BLGmx_corr2(y,ypred).^2;
 RSQ=frsq(Ytest,Pred);
end
display(['layer',num2str(layi)]);
end

%%
% giordano Kell 
% pool1 size: 1  96  43  43  80
% pool2 size: 1  256   11   11   80
% conv3 size: 1  512   11   11   80
% conv4_W size: 1  1024    11    11    80
% pool5_flat_W size: 1  18432      1      1     80
% fc6_W size: 1  1024     1     1    80
% conv4_G size: 1  1024    11    11    80
% pool5_flat_G size: 1  18432      1      1     80
% fc6_G size: 1  1024     1     1    80
% 
% 
% giordano VGGish 
% pool1 size: 2  64  32  48  80
% pool2 size: 2  128   16   24   80
% pool3 size: 2  256    8   12   80
% pool4 size: 2  512    4    6   80
% fc1_1 size: 2  4096     1     1    80
% fc1_2 size: 2  4096     1     1    80
% fc2 size: 2  128    1    1   80
% 
% 
% giordano Yamnet 
% layer01relu size: 2  32  32  48  80
% layer02relu size: 2  64  32  48  80
% layer03relu size: 2  128   16   24   80
% layer04relu size: 2  128   16   24   80
% layer05relu size: 2  256    8   12   80
% layer06relu size: 2  256    8   12   80
% layer07relu size: 2  512    4    6   80
% layer08relu size: 2  512    4    6   80
% layer09relu size: 2  512    4    6   80
% layer10relu size: 2  512    4    6   80
% layer11relu size: 2  512    4    6   80
% layer12relu size: 2  512    4    6   80
% layer13relu size: 2  1024     2     3    80
% layer14relu size: 2  1024     2     3    80
% 
% 
% formisano Kell
% pool1 size: 1   96   43   43  288
% pool2 size: 1  256   11   11  288
% conv3 size: 1  512   11   11  288
% conv4_W size: 1  1024    11    11   288
% pool5_flat_W size: 1  18432      1      1    288
% fc6_W size: 1  1024     1     1   288
% conv4_G size: 1  1024    11    11   288
% pool5_flat_G size: 1  18432      1      1    288
% fc6_G size: 1  1024     1     1   288
% 
% 
% formisano VGGish
% pool1 size: 1   64   32   48  288
% pool2 size: 1  128   16   24  288
% pool3 size: 1  256    8   12  288
% pool4 size: 1  512    4    6  288
% fc1_1 size: 1  4096     1     1   288
% fc1_2 size: 1  4096     1     1   288
% fc2 size: 1  128    1    1  288
% 
% 
% formisano Yamnet
% layer01relu size: 1   32   32   48  288
% layer02relu size: 1   64   32   48  288
% layer03relu size: 1  128   16   24  288
% layer04relu size: 1  128   16   24  288
% layer05relu size: 1  256    8   12  288
% layer06relu size: 1  256    8   12  288
% layer07relu size: 1  512    4    6  288
% layer08relu size: 1  512    4    6  288
% layer09relu size: 1  512    4    6  288
% layer10relu size: 1  512    4    6  288
% layer11relu size: 1  512    4    6  288
% layer12relu size: 1  512    4    6  288
% layer13relu size: 1  1024     2     3   288
% layer14relu size: 1  1024     2     3   320