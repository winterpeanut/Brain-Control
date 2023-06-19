naturalpath = ['D:\EXP2\AcoSemDNN_Behav_fMRI_Repo\',...
    'AcoSemDNN_Behav_fMRI_Repo\data\formisano_acoustics\']
MTFpath = [naturalpath,'formisano_MTF.mat'];
CGpath = [naturalpath,'formisano_cochleagram.mat'];

load(MTFpath,'MTF_TMabs')
load(CGpath,'CG_TMabs')



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
fs = 16000;
loadload;
paras = [8, 8, -2, log2(fs/16000)];
para1 = [paras];%,0,0,1]
rv = 2.^(2:1:8);
sv = 2.^(-2:.5:3);
avertime = 1;
nsyn = 5;
synlayer = 'conv4_1';
filepx = [pattern,'_',Starter,'_',Betastr,'_',num2str(iters),...
    '_',num2str(r_lamb),'rms_',num2str(l_lamb),...
    'lv_subj',num2str(subj),'_',roi,'_',method];
wd = [md,pattern,filesep,synlayer,'\audio\'];

clusters{1} = [2,5,4,3,4];%speech
clusters{2} = [5,0,1,5,2];%music
vecsyn_music = [];vecsyn_speech=[];vecsyn_music_cg = [];vecsyn_speech_cg=[];
vecsyn_musictmp = [];vecsyn_speechtmp=[];vecsyn_music_cgtmp = [];vecsyn_speech_cgtmp=[];
whichsubj = [1,3,4];
for subj = whichsubj
    filepx = [pattern,'_',Starter,'_',Betastr,'_',num2str(iters),'_',num2str(r_lamb),...
        'rms_',num2str(l_lamb),'lv_subj',num2str(subj),'_',roi,'_',method];
    
    for i = 1:nsyn
        
        % music
        synsound = [];
        filesx2 = ['_cluster',num2str(clusters{2}(subj)),'_6comps_waveform_SV10',...
            dospeech,num2str(i),'.wav'];
        waveform2 = [wd,filepx,filesx2];
        [synsound,~] = audioread(waveform2);
        synsound=synsound-mean(synsound);
        synsound=unitseq(synsound);
        spec = wav2aud(synsound,paras);
        cr = aud2cor(spec, para1, rv, sv,'tmpxxx',0);
        MTF_syn_music = permute(cr,[3,4,1,2]);
        if avertime ==1
            MTF_syn_music = abs(mean(MTF_syn_music,1));
            vecsyn_musictmp(:,i) = reshape(MTF_syn_music,[],1);
            vecsyn_music_cgtmp(:,i) = reshape(mean(spec,1),[],1);
        else
            MTF_syn_music = abs(MTF_syn_music);
            vecsyn_musictmp(:,i) = reshape(MTF_syn_music,[],1);
            vecsyn_music_cgtmp(:,i) = reshape(spec,[],1);
        end
        
        %speech
        synsound = [];
        filesx = ['_cluster',num2str(clusters{1}(subj)),'_6comps_waveform_SV10'...
            ,dospeech,num2str(i),'.wav'];
        waveform = [wd,filepx,filesx];
        [synsound,~] = audioread(waveform);
        synsound=synsound-mean(synsound);
        synsound=unitseq(synsound);
        spec = wav2aud(synsound,paras);
        cr = aud2cor(spec, para1, rv, sv,'tmpxxx',0);
        MTF_syn_speech = permute(cr,[3,4,1,2]);
        if avertime == 1
            MTF_syn_speech = abs(mean(MTF_syn_speech,1));
            vecsyn_speechtmp(:,i) = reshape(MTF_syn_speech,[],1);
            vecsyn_speech_cgtmp(:,i) = reshape(mean(spec,1),[],1);
        else
            MTF_syn_speech = abs(MTF_syn_speech);
            vecsyn_speechtmp(:,i) = reshape(MTF_syn_speech,[],1);
            vecsyn_speech_cgtmp(:,i) = reshape(spec,[],1);
        end
    end
    vecsyn_music = cat(2,vecsyn_music,vecsyn_musictmp);
    vecsyn_speech = cat(2,vecsyn_speech,vecsyn_speechtmp);
    vecsyn_music_cg = cat(2,vecsyn_music_cg,vecsyn_music_cgtmp);
    vecsyn_speech_cg = cat(2,vecsyn_speech_cg,vecsyn_speech_cgtmp);
end
%%
Betastr = 'randbeta';
iters = 800;
nsyn = 5;
synlayer = 'conv4_1';
filepx = [pattern,'_',Starter,'_',Betastr,'_',num2str(iters),...
    '_',num2str(r_lamb),'rms_',num2str(l_lamb),...
    'lv_subj',num2str(subj),'_',roi,'_',method];

vecsyn_musictmp = [];vecsyn_speechtmp=[];vecsyn_music_cgtmp = [];vecsyn_speech_cgtmp=[];
whichsubj = [1,3,4];
for subj = whichsubj
    filepx = [pattern,'_',Starter,'_',Betastr,'_',num2str(iters),'_',num2str(r_lamb),...
        'rms_',num2str(l_lamb),'lv_subj',num2str(subj),'_',roi,'_',method];
    
    for i = 1:nsyn
        
        % music
        synsound = [];
        filesx2 = ['_cluster',num2str(clusters{2}(subj)),'_6comps_waveform_SV10',...
            dospeech,num2str(i),'.wav'];
        waveform2 = [wd,filepx,filesx2];
        [synsound,~] = audioread(waveform2);
        synsound=synsound-mean(synsound);
        synsound=unitseq(synsound);
        spec = wav2aud(synsound,paras);
        cr = aud2cor(spec, para1, rv, sv,'tmpxxx',0);
        MTF_syn_music = permute(cr,[3,4,1,2]);
        if avertime ==1
            MTF_syn_music = abs(mean(MTF_syn_music,1));
            vecsyn_musictmp(:,i) = reshape(MTF_syn_music,[],1);
            vecsyn_music_cgtmp(:,i) = reshape(mean(spec,1),[],1);
        else
            MTF_syn_music = abs(MTF_syn_music);
            vecsyn_musictmp(:,i) = reshape(MTF_syn_music,[],1);
            vecsyn_music_cgtmp(:,i) = reshape(spec,[],1);
        end
        
        %speech
        synsound = [];
        filesx = ['_cluster',num2str(clusters{1}(subj)),'_6comps_waveform_SV10'...
            ,dospeech,num2str(i),'.wav'];
        waveform = [wd,filepx,filesx];
        [synsound,~] = audioread(waveform);
        synsound=synsound-mean(synsound);
        synsound=unitseq(synsound);
        spec = wav2aud(synsound,paras);
        cr = aud2cor(spec, para1, rv, sv,'tmpxxx',0);
        MTF_syn_speech = permute(cr,[3,4,1,2]);
        if avertime == 1
            MTF_syn_speech = abs(mean(MTF_syn_speech,1));
            vecsyn_speechtmp(:,i) = reshape(MTF_syn_speech,[],1);
            vecsyn_speech_cgtmp(:,i) = reshape(mean(spec,1),[],1);
        else
            MTF_syn_speech = abs(MTF_syn_speech);
            vecsyn_speechtmp(:,i) = reshape(MTF_syn_speech,[],1);
            vecsyn_speech_cgtmp(:,i) = reshape(spec,[],1);
        end
    end
    vecsyn_music = cat(2,vecsyn_music,vecsyn_musictmp);
    vecsyn_speech = cat(2,vecsyn_speech,vecsyn_speechtmp);
    vecsyn_music_cg = cat(2,vecsyn_music_cg,vecsyn_music_cgtmp);
    vecsyn_speech_cg = cat(2,vecsyn_speech_cg,vecsyn_speech_cgtmp);
end

%%
catspeech = 1;
idxspeech = 48*(catspeech-1)+1:48*catspeech;
catmusic = 4;
idxmusic = 48*(catmusic-1)+1:48*catmusic;
catvoice = 2;
idxvoice = 48*(catvoice-1)+1:48*catvoice;
idxbsm =[idxspeech,idxmusic];

MTF_natural = reshape(MTF_TMabs,[],288);
CG_natural = reshape(CG_TMabs,[],288);
whichmodel = 'CG';
newmatrix = [];
if strcmp(whichmodel,'MTF')
    newmatrix = cat(2,MTF_natural,vecsyn_speech,vecsyn_music);
else
    newmatrix = cat(2,CG_natural,vecsyn_speech_cg,vecsyn_music_cg);
end
%newmatrix = MTF_natural;
n = size(newmatrix,2);
pairs = nchoosek(1:n,2);%48*2+2
npairs = size(pairs,1);
clear newdis
parfor i =1 : npairs
    %    voice2voice(i) = pdist([vecvoice(:,pairs(i,1)) ,vecvoice(:,pairs(i,2)) ]','cos');
    %    music2music(i) = pdist([vecmusic(:,pairs(i,1)) ,vecmusic(:,pairs(i,2))]','cos');
    newdis(i) = pdist([newmatrix(:,pairs(i,1)) ,newmatrix(:,pairs(i,2))]','cos');
end
%%
w = true(n);
b = zeros(n);
b (tril(w,-1)) = newdis;
b = b+b';
%imagesc(b);
Y = mdscale(b,2);
%figure;box off;
RGB = getColor;
figure;box off;
RGB = getColor;
natnams={'speech','voice','animal','music','nature','tool','syn-speech','syn-music','rand-speech','rand-music'};
hold on;
for i = 1:6
    idx = 48*(i-1)+1:48*i;
    h(i) = scatter(Y(idx,1),Y(idx,2),'fill','MarkerFaceColor',RGB{i},'MarkerEdgeColor',[0,0,0]);
end
ns=288;
nsub = length(whichsubj);
 Betastr ='both'

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
%% calculate the dist to centroid


%%
yaxr  = [0.1,0.6];
music2aver =[];speech2aver = [];
for j = 1:nsyn*nsub
    music1 = vecsyn_music(:,j);
    speech1 = vecsyn_speech(:,j);
    music2voice=[];music2music=[];voice2voice=[];voice2music=[];
    for i = 1:288
        if strcmp(whichmodel,'MTF')
            nat = mean(MTF_natural(:,i),2);
        else
            nat = mean(CG_natural(:,i),2);
            music1 = vecsyn_music_cg(:,j);
            speech1 = vecsyn_speech_cg(:,j);
        end
       
        music2aver(i,j) = pdist([music1,nat]','cos');
        speech2aver(i,j) = pdist([speech1,nat]','cos');
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
%[h,p] = ttest(mean(music2music,2),mean(voice2music,2))
%bar([1,2],[mean(voice2voice),mean(music2music)])
if strcmp(Betastr,'both')
    music2aver =[];speech2aver = [];music2averR =[];speech2averR = [];
    for j = 1:nsyn*nsub      
        music2 = vecsyn_music(:,j+nsyn*nsub);
        speech2 = vecsyn_speech(:,j+nsyn*nsub);       
        music2voice=[];music2music=[];voice2voice=[];voice2music=[];
        for i = 1:288
            if strcmp(whichmodel,'MTF')
             nat = mean(MTF_natural(:,i),2);
            else
             nat = mean(CG_natural(:,i),2);
             music2 = vecsyn_music_cg(:,j+nsyn*nsub);
             speech2 = vecsyn_speech_cg(:,j+nsyn*nsub);
            end
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