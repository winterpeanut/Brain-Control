
datapath = 'D:\EXP2\fMRIdata\vtcs\';
subj = 1;
sess = 1;
sounds = [];silence = [];vtcall = [];
for run = 1:6
datanams = [datapath,'subj',num2str(subj),'_sess',num2str(sess),'_run',num2str(run),'.vtc'];
mrknams = [datapath,'sess',num2str(sess),'_run',num2str(run),'.prt'];
VTC = xff(datanams);
MRK = xff(mrknams);

mrk = squeeze(struct2cell(MRK.Cond));
mrk = mrk(3,:);
mrksil = mrk{end}(:,1)+1;
allmrksil{run} = mrksil; 
mrksil(mrksil>size(vtc,1))=[];

mrksounds = cell2mat(arrayfun(@(x) mrk{x}(:,1),[1:72],'UniformOutput',false))+1;
vtc = VTC.VTCData;
vtc = double(reshape(vtc,size(vtc,1),[]));
sounds = cat(3,sounds,vtc(mrksounds,:));
silence = cat(3,silence,vtc(mrksil,:));

end
soundsavg = mean(sounds,3);
silenceavg = mean(silence,3);

idx = cell2mat(arrayfun(@(x)any(soundsavg(:,x)~=0),[1:size(sounds,2)],'UniformOutput',false));
soundsavg = soundsavg(:,idx)';
silenceavg = silenceavg(:,idx)';


soundscell = mat2cell(soundsavg,ones(size(soundsavg,1),1));
silencecell = mat2cell(silenceavg,ones(size(soundsavg,1),1));

[h,p] = cellfun(@(x,y)ttest2(x,y,'Tail','both'),soundscell,silencecell,'UniformOutput',false);
p = cell2mat(p);
numel(p(p<0.05))
%%



v12 = permute(mean(alldata(:,:,1:2),3),[1,3,2]);
v34 = permute(mean(alldata(:,:,3:4),3),[1,3,2]);
v12 = rand(288,1)
v34 = rand(288,1)

p = mtimesx((mtimesx(v34,'t',v12,'n') ./sum(v34.^2,1)),v34);

r = squeeze(1- sqrt(sum((v12-p).^2))./sqrt(sum(v12.^2)));

corr(v12,v34)