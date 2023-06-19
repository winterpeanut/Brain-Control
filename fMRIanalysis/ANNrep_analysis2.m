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
md =  'D:\EXP2\Results\DirectlyWaveform\';
dospeech = '_dospeech';
pattern = 'counter';
Betastr = 'realbeta';
Starter = 'randstarter';
iters = 800;
r_lamb = 1;
l_lamb = 0.01;

roi = 'allroi';
method = 'Decomp';
layer = 'conv4_1';
synlayer = 'conv4_1';
nsyn = 18;
wd = [md,pattern,filesep,synlayer];
filepath = [wd,'\'];
cluster_speech = [2,5,4,3,4];
cluster_music = [5,0,1,5,2];

tmpmusic = [];tmpspeech=[];
vecsyn_music=[];vecsyn_speech=[];
vecsyn_musictmp=[];vecsyn_speechtmp=[];
vecsyn_musicR=[];vecsyn_speechR=[];
vecsyn_musictmpR=[];vecsyn_speechtmpR=[];
whichsubj = [1,3,4];

for subj = whichsubj
    if ~strcmp(Betastr,'both')
        filepx = [pattern,'_',Starter,'_',Betastr,'_',num2str(iters),...
            '_',num2str(r_lamb),'rms_',num2str(l_lamb),...
            'lv_subj',num2str(subj),'_',roi,'_',method];
    else
        filepx = [pattern,'_',Starter,'_realbeta_',num2str(iters),...
            '_',num2str(r_lamb),'rms_',num2str(l_lamb),...
            'lv_subj',num2str(subj),'_',roi,'_',method];
    end
    filesx = ['_cluster',num2str(cluster_speech(subj)),'_6comps_waveform_SV10',dospeech,'.hdf5'];
    waveform = [filepath,filepx,filesx];
    tmpspeech=h5read(waveform,['/',layer]);
    
    filesx2 = ['_cluster',num2str(cluster_music(subj)),'_6comps_waveform_SV10',dospeech,'.hdf5'];
    waveform = [filepath,filepx,filesx2];
    tmpmusic=h5read(waveform,['/',layer]);
    
    if ndims(tmpmusic)==4
        vecsyn_musictmp = reshape(mean(tmpmusic,3),[],nsyn);
        vecsyn_speechtmp = reshape(mean(tmpspeech,3),[],nsyn);
    else
        vecsyn_musictmp = tmpmusic;
        vecsyn_speechtmp = tmpspeech;
    end
    vecsyn_music = cat(2,vecsyn_music,vecsyn_musictmp);
    vecsyn_speech = cat(2,vecsyn_speech,vecsyn_speechtmp);
end%

if strcmp(Betastr,'both')
    for subj = whichsubj
        filepx2 = [pattern,'_',Starter,'_randbeta_',num2str(iters),...
            '_',num2str(r_lamb),'rms_',num2str(l_lamb),...
            'lv_subj',num2str(subj),'_',roi,'_',method];
        
        filesx = ['_cluster',num2str(cluster_speech(subj)),'_6comps_waveform_SV10',dospeech,'.hdf5'];
        filesx2 = ['_cluster',num2str(cluster_music(subj)),'_6comps_waveform_SV10',dospeech,'.hdf5'];
        
        waveform2 = [filepath,filepx2,filesx2];
        tmpmusicR=h5read(waveform2,['/',layer]);
        waveform = [filepath,filepx2,filesx];
        tmpspeechR=h5read(waveform,['/',layer]);
        
        if ndims(tmpmusicR)==4
            vecsyn_musictmpR = reshape(mean(tmpmusicR,3),[],nsyn);
            vecsyn_speechtmpR = reshape(mean(tmpspeechR,3),[],nsyn);
        else
            vecsyn_musictmpR = tmpmusicR;
            vecsyn_speechtmpR = tmpspeechR;
        end
        vecsyn_musicR = cat(2,vecsyn_musicR,vecsyn_musictmpR);
        vecsyn_speechR = cat(2,vecsyn_speechR,vecsyn_speechtmpR);
        vecsyn_music = cat(2,vecsyn_music,vecsyn_musictmpR);
        vecsyn_speech = cat(2,vecsyn_speech,vecsyn_speechtmpR);
    end%
end
%%
distmethod = 'cos';
catspeech = 1;
idxspeech = 48*(catspeech-1)+1:48*catspeech;
catmusic = 4;
idxmusic = 48*(catmusic-1)+1:48*catmusic;
catvoice = 2;
idxvoice = 48*(catvoice-1)+1:48*catvoice;
idxbsm =[idxspeech,idxmusic];

idxlayer = strcmp(layer_nams,layer);
newmatrix = [];
newmatrix = cat(2,Reps{idxlayer}(:,:),vecsyn_speech,vecsyn_music);
n = size(newmatrix,2);
pairs = nchoosek(1:n,2);%48*2+2
npairs = size(pairs,1);
clear newdis
parfor i =1 : npairs
    newdis(i) = pdist([newmatrix(:,pairs(i,1)) ,newmatrix(:,pairs(i,2)) ]',distmethod);
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
natnams={'speech','voice','animal','music','nature','tool','syn-speech','syn-music','rand-speech','rand-music'};
hold on;
for i = 1:6
    idx = 48*(i-1)+1:48*i;
    h(i) = scatter(Y(idx,1),Y(idx,2),'fill','MarkerFaceColor',RGB{i},'MarkerEdgeColor',[0,0,0]);
end
ns=288;
nsub = length(whichsubj);

if strcmp(Betastr,'both')
    idxsyn = ns+1:ns+4*nsyn*nsub;
    idxsyn = reshape(idxsyn,nsyn*nsub,4)';
    h(7) = scatter(Y(idxsyn(1,:),1),Y(idxsyn(1,:),2),'MarkerEdgeColor',RGB{1},'LineWidth',2);
    h(9) = scatter(Y(idxsyn(3,:),1),Y(idxsyn(3,:),2),'MarkerEdgeColor',RGB{4},'LineWidth',2);
    h(8) = scatter(Y(idxsyn(2,:),1),Y(idxsyn(2,:),2),'x','MarkerEdgeColor',RGB{1},'LineWidth',2);
    h(10) = scatter(Y(idxsyn(4,:),1),Y(idxsyn(4,:),2),'x','MarkerEdgeColor',RGB{4},'LineWidth',2);
    legend([h([1:10])],natnams{[1:6,7,9,8,10]});box off;
else
    idxsyn = ns+1:ns+2*nsyn*nsub;
    idxsyn = reshape(idxsyn,nsyn*nsub,2)';
    h(7) = scatter(Y(idxsyn(1,:),1),Y(idxsyn(1,:),2),'MarkerEdgeColor',RGB{1},'LineWidth',2);
    h(8) = scatter(Y(idxsyn(2,:),1),Y(idxsyn(2,:),2),'MarkerEdgeColor',RGB{4},'LineWidth',2);
    legend([h([1:8])],natnams{[1:8]});box off;
end


%%
yaxr = [0.5,1];

idxlayer = find(strcmp(layer_nams,layer));
music2aver =[];speech2aver = [];music2averR =[];speech2averR = [];
for j = 1:nsyn*nsub
    music1 = vecsyn_music(:,j);
    speech1 = vecsyn_speech(:,j);
    music2voice=[];music2music=[];voice2voice=[];voice2music=[];
    for i = 1:288
        nat = mean(Reps{idxlayer}(:,i),2);
        
        music2aver(i,j) = pdist([music1,nat]',distmethod);
        speech2aver(i,j) = pdist([speech1,nat]',distmethod);
      
    end
end
music2aver = reshape(permute(reshape(music2aver,48,6,[]),[1,3,2]),[],6);
speech2aver = reshape(permute(reshape(speech2aver,48,6,[]),[1,3,2]),[],6);

figure;
subplot 121
hold on;
mm = mean(speech2aver,1);
error = std(speech2aver,0,1)./sqrt(nsyn);
bar([1:6],mm);xticks([1:6]);xticklabels(natnams(1:6));
errorbar([1:6],mm,error,'|k','LineWidth',2);
ylim([yaxr])

subplot 122
hold on;
mm = mean(music2aver,1);
error = std(music2aver,0,1)./sqrt(nsyn);
bar([1:6],mm);xticks([1:6]);xticklabels(natnams(1:6));
errorbar([1:6],mm,error,'|k','LineWidth',2);
ylim([yaxr])

if strcmp(Betastr,'both')
    music2aver =[];speech2aver = [];music2averR =[];speech2averR = [];
    for j = 1:nsyn*nsub
        
        music2 = vecsyn_musicR(:,j);
        speech2 = vecsyn_speechR(:,j);
        
        music2voice=[];music2music=[];voice2voice=[];voice2music=[];
        for i = 1:288
            nat = mean(Reps{idxlayer}(:,i),2);
            music2averR(i,j) = pdist([music2,nat]',distmethod);
            speech2averR(i,j) = pdist([speech2,nat]',distmethod);
        end
    end
    
    music2averR = reshape(permute(reshape(music2averR,48,6,[]),[1,3,2]),[],6);
    speech2averR = reshape(permute(reshape(speech2averR,48,6,[]),[1,3,2]),[],6);
    
    figure;
    subplot 121
    hold on;
    mm = mean(speech2averR,1);
    error = std(speech2averR,0,1)./sqrt(nsyn);
    bar([1:6],mm);xticks([1:6]);xticklabels(natnams(1:6));
    errorbar([1:6],mm,error,'|k','LineWidth',2);
    ylim([yaxr])
    
    subplot 122
    hold on;
    mm = mean(music2averR,1);
    error = std(music2averR,0,1)./sqrt(nsyn);
    bar([1:6],mm);xticks([1:6]);xticklabels(natnams(1:6));
    errorbar([1:6],mm,error,'|k','LineWidth',2);
    ylim([yaxr])
end
%[h,p] = ttest(mean(music2music,2),mean(voice2music,2))
%bar([1,2],[mean(voice2voice),mean(music2music)])