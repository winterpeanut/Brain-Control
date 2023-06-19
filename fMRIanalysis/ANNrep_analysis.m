load('D:\EXP2\AcoSemDNN_Behav_fMRI_Repo\AcoSemDNN_Behav_fMRI_Repo\data\formisano_dnns\vggishrandom\embedding_randomInit_VGGish_randomWeights.mat')

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
    
    dat_rand_tmp = double(eval(['randomInit_model_layers.',layer_nams{j}]));  
    dat_rand_tmp = permute(dat_rand_tmp,[2,3,4,5,1]);

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
       dat_rand_tmp = mean(dat_rand_tmp);
   end
   s=size(dat_tmp);
   thisdat_tmp=reshape(dat_tmp,[prod(s(1:4)) s(5)]);
   thisdat_rand_tmp=reshape(dat_rand_tmp,[prod(s(1:4)) s(5)]);

   Reps{j,1} = thisdat_tmp;
   Reps_rand{j,1} = thisdat_rand_tmp;

  % Reps{1,1} = cat(1,Reps{1,1},thisdat_tmp);
   %         RDMsEuc{end+1,1}=BLG_EucDistND(thisdat_tmp);
   %         RDMsCos{end+1,1}=BLG_CosDistND(thisdat_tmp);
   Components{end+1,1}=[layer_nams{j}];
   Ndims=cat(1,Ndims,size(thisdat_tmp,1));
end

%%
catvoice = 1;
idxvoice = 48*(catvoice-1)+1:48*catvoice;
catmusic = 4;
idxmusic = 48*(catmusic-1)+1:48*catmusic;

MTF_voice = MTF_TMabs(:,:,:,:,idxvoice);
MTF_music= MTF_TMabs(:,:,:,:,idxmusic);

%%
md =  'D:\EXP2\Results\DirectlyWaveform\';
dospeech = '_dospeech';
pattern = 'stretch';
Betastr = 'realbeta';
Starter = 'randstarter';
iters = 800;
r_lamb = 1;
l_lamb = 0.01;
subj = 1;
roi = 'allroi';
method = 'Decomp';
layer = 'conv4_1';
synlayer = 'conv'
tmpmusic = [];tmpspeech=[];vecsyn_music=[];vecsyn_speech=[];
wd = [md,pattern];
filepath = [wd,'\'];
for i=1:4
cluster = 2;
filepx = [pattern,'_',Starter,'_',Betastr,'_',num2str(iters),...
    '_',num2str(r_lamb),'rms_',num2str(l_lamb),...
    'lv_subj',num2str(subj),'_',roi,'_',method];

filesx2 = ['_cluster',num2str(cluster),'_6comps_',num2str(i),'_waveform_SV10',dospeech,'.hdf5'];
waveform = [filepath,filepx,filesx2];

tmpspeech=h5read(waveform,['/',layer]);

cluster = 5;
filepx = [pattern,'_',Starter,'_',Betastr,'_',num2str(iters),...
    '_',num2str(r_lamb),'rms_',num2str(l_lamb),...
    'lv_subj',num2str(subj),'_',roi,'_',method];
wd = [md,pattern];

filesx2 = ['_cluster',num2str(cluster),'_6comps_',num2str(i),'_waveform_SV10',dospeech,'.hdf5'];
waveform = [filepath,filepx,filesx2];

tmpmusic=h5read(waveform,['/',layer]);

if ndims(tmpmusic)==3
    vecsyn_music(:,i) = reshape(mean(tmpmusic,3),[],1);
    vecsyn_speech(:,i) = reshape(mean(tmpspeech,3),[],1);
else
    vecsyn_music(:,i) = tmpmusic;
    vecsyn_speech(:,i) = tmpspeech;
end
end
%




%%
catspeech = 1;
idxspeech = 48*(catspeech-1)+1:48*catspeech;
catmusic = 4;
idxmusic = 48*(catmusic-1)+1:48*catmusic;
catvoice = 2;
idxvoice = 48*(catvoice-1)+1:48*catvoice;
idxbsm =[idxspeech,idxmusic];

idxlayer = find(strcmp(layer_nams,layer));
newmatrix = [];
newmatrix = cat(2,Reps{idxlayer}(:,:),vecsyn_speech,vecsyn_music);
n = size(newmatrix,2);
pairs = nchoosek(1:n,2);%48*2+2
npairs = size(pairs,1);
clear newdis
for i =1 : npairs
%    voice2voice(i) = pdist([vecvoice(:,pairs(i,1)) ,vecvoice(:,pairs(i,2)) ]','cos');
%    music2music(i) = pdist([vecmusic(:,pairs(i,1)) ,vecmusic(:,pairs(i,2))]','cos');  
   newdis(i) = pdist([newmatrix(:,pairs(i,1)) ,newmatrix(:,pairs(i,2)) ]','cos');
end
%%
w = true(n);
b = zeros(n);
b (tril(w,-1)) = newdis;
b = b+b';
%imagesc(b);
Y = mdscale(b,2);
figure;
RGB = getColor;
natnams={'speech','voice','animal','music','nature','tool','syn-speech','syn-music'};
hold on;
for i = 1:6
    idx = 48*(i-1)+1:48*i;
    h(i) = scatter(Y(idx,1),Y(idx,2),'fill','MarkerFaceColor',RGB{i},'MarkerEdgeColor',[0,0,0]);
end
ns=288;
nsyn = 4;
idxsyn = [ns+1:ns+nsyn;ns+nsyn+1:ns+2*nsyn];
    h(7) = scatter(Y(idxsyn(1,:),1),Y(idxsyn(1,:),2),'MarkerEdgeColor',RGB{1},'LineWidth',2);
    h(8) = scatter(Y(idxsyn(2,:),1),Y(idxsyn(2,:),2),'MarkerEdgeColor',RGB{4},'LineWidth',2);
legend([h([1:8])],natnams{[1:8]});box off;
%%


idxlayer = find(strcmp(layer_nams,layer));

for j = 1:4
music1 = vecsyn_music(:,j);
speech1 = vecsyn_speech(:,j);
 music2voice=[];music2music=[];voice2voice=[];voice2music=[];
for i = 1:6
    idxcat= 48*(i-1)+1:48*i;
    nat = mean(Reps{idxlayer}(:,idxcat),2);
%    music2voice(i,j) = pdist([vecsyn_music,vecvoice(:,i)]','cos');
%    music2music(i,j) = pdist([vecsyn_music,vecmusic(:,i)]','cos'); 
%   
%    voice2voice(i,j) = pdist([vecsyn_speech,vecvoice(:,i)]','cos');
%    voice2music(i,j) = pdist([vecsyn_speech,vecmusic(:,i)]','cos'); 
    
   music2aver(i,j) = pdist([music1,nat]','cos');
   speech2aver(i,j) = pdist([speech1,nat]','cos');
end
end
figure;
subplot 211
bar([1:6],mean(speech2aver,2));xticks([1:6]);xticklabels(natnams(1:6));
subplot 212
bar([1:6],mean(music2aver,2));xticks([1:6]);xticklabels(natnams(1:6));
%[h,p] = ttest(mean(music2music,2),mean(voice2music,2))
%bar([1,2],[mean(voice2voice),mean(music2music)])