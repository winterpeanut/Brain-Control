
datapath = 'D:\EXP2\fMRIdata';
load([datapath,filesep,'Beta2hem.mat'],'HG2hem','PP2hem','PT2hem','mSTG2hem','aSTG2hem','pSTG2hem')
HGtest(:,1,1) = mean(HG2hem.CV1.KV.BetasTest,2); % n_stim, n_fold
HGtest(:,1,2) = mean(HG2hem.CV2.KV.BetasTest,2); 
HGtest(:,1,3) = mean(HG2hem.CV3.KV.BetasTest,2); 
HGtest(:,1,4) = mean(HG2hem.CV4.KV.BetasTest,2); 

HGtrain(:,1,1) = mean(HG2hem.CV1.KV.BetasTrain,2); % n_stim, n_fold
HGtrain(:,1,2) = mean(HG2hem.CV2.KV.BetasTrain,2); 
HGtrain(:,1,3) = mean(HG2hem.CV3.KV.BetasTrain,2); 
HGtrain(:,1,4) = mean(HG2hem.CV4.KV.BetasTrain,2); 

%
PPtest(:,1,1) = mean(PP2hem.CV1.KV.BetasTest,2); % n_stim, n_fold
PPtest(:,1,2) = mean(PP2hem.CV2.KV.BetasTest,2); 
PPtest(:,1,3) = mean(PP2hem.CV3.KV.BetasTest,2); 
PPtest(:,1,4) = mean(PP2hem.CV4.KV.BetasTest,2); 

PPtrain(:,1,1) = mean(PP2hem.CV1.KV.BetasTrain,2); % n_stim, n_fold
PPtrain(:,1,2) = mean(PP2hem.CV2.KV.BetasTrain,2); 
PPtrain(:,1,3) = mean(PP2hem.CV3.KV.BetasTrain,2); 
PPtrain(:,1,4) = mean(PP2hem.CV4.KV.BetasTrain,2);

%
PTtest(:,1,1) = mean(PT2hem.CV1.KV.BetasTest,2); % n_stim, n_fold
PTtest(:,1,2) = mean(PT2hem.CV2.KV.BetasTest,2); 
PTtest(:,1,3) = mean(PT2hem.CV3.KV.BetasTest,2); 
PTtest(:,1,4) = mean(PT2hem.CV4.KV.BetasTest,2); 

PTtrain(:,1,1) = mean(PT2hem.CV1.KV.BetasTrain,2); % n_stim, n_fold
PTtrain(:,1,2) = mean(PT2hem.CV2.KV.BetasTrain,2); 
PTtrain(:,1,3) = mean(PT2hem.CV3.KV.BetasTrain,2); 
PTtrain(:,1,4) = mean(PT2hem.CV4.KV.BetasTrain,2); 

%
mSTGtest(:,1,1) = mean(mSTG2hem.CV1.KV.BetasTest,2); % n_stim, n_fold
mSTGtest(:,1,2) = mean(mSTG2hem.CV2.KV.BetasTest,2); 
mSTGtest(:,1,3) = mean(mSTG2hem.CV3.KV.BetasTest,2); 
mSTGtest(:,1,4) = mean(mSTG2hem.CV4.KV.BetasTest,2); 

mSTGtrain(:,1,1) = mean(mSTG2hem.CV1.KV.BetasTrain,2); % n_stim, n_fold
mSTGtrain(:,1,2) = mean(mSTG2hem.CV2.KV.BetasTrain,2); 
mSTGtrain(:,1,3) = mean(mSTG2hem.CV3.KV.BetasTrain,2); 
mSTGtrain(:,1,4) = mean(mSTG2hem.CV4.KV.BetasTrain,2); 

%
aSTGtest(:,1,1) = mean(aSTG2hem.CV1.KV.BetasTest,2); % n_stim, n_fold
aSTGtest(:,1,2) = mean(aSTG2hem.CV2.KV.BetasTest,2); 
aSTGtest(:,1,3) = mean(aSTG2hem.CV3.KV.BetasTest,2); 
aSTGtest(:,1,4) = mean(aSTG2hem.CV4.KV.BetasTest,2); 

aSTGtrain(:,1,1) = mean(aSTG2hem.CV1.KV.BetasTrain,2); % n_stim, n_fold
aSTGtrain(:,1,2) = mean(aSTG2hem.CV2.KV.BetasTrain,2); 
aSTGtrain(:,1,3) = mean(aSTG2hem.CV3.KV.BetasTrain,2); 
aSTGtrain(:,1,4) = mean(aSTG2hem.CV4.KV.BetasTrain,2); 

%
pSTGtest(:,1,1) = mean(pSTG2hem.CV1.KV.BetasTest,2); % n_stim, n_fold
pSTGtest(:,1,2) = mean(pSTG2hem.CV2.KV.BetasTest,2); 
pSTGtest(:,1,3) = mean(pSTG2hem.CV3.KV.BetasTest,2); 
pSTGtest(:,1,4) = mean(pSTG2hem.CV4.KV.BetasTest,2); 

pSTGtrain(:,1,1) = mean(pSTG2hem.CV1.KV.BetasTrain,2); % n_stim, n_fold
pSTGtrain(:,1,2) = mean(pSTG2hem.CV2.KV.BetasTrain,2); 
pSTGtrain(:,1,3) = mean(pSTG2hem.CV3.KV.BetasTrain,2); 
pSTGtrain(:,1,4) = mean(pSTG2hem.CV4.KV.BetasTrain,2); 
%%

HGtest(:,2,1) = mean(HG2hem.CV1.RS.BetasTest,2); % n_stim, n_fold
HGtest(:,2,2) = mean(HG2hem.CV2.RS.BetasTest,2); 
HGtest(:,2,3) = mean(HG2hem.CV3.RS.BetasTest,2); 
HGtest(:,2,4) = mean(HG2hem.CV4.RS.BetasTest,2); 

HGtrain(:,2,1) = mean(HG2hem.CV1.RS.BetasTrain,2); % n_stim, n_fold
HGtrain(:,2,2) = mean(HG2hem.CV2.RS.BetasTrain,2); 
HGtrain(:,2,3) = mean(HG2hem.CV3.RS.BetasTrain,2); 
HGtrain(:,2,4) = mean(HG2hem.CV4.RS.BetasTrain,2); 

%
PPtest(:,2,1) = mean(PP2hem.CV1.RS.BetasTest,2); % n_stim, n_fold
PPtest(:,2,2) = mean(PP2hem.CV2.RS.BetasTest,2); 
PPtest(:,2,3) = mean(PP2hem.CV3.RS.BetasTest,2); 
PPtest(:,2,4) = mean(PP2hem.CV4.RS.BetasTest,2); 

PPtrain(:,2,1) = mean(PP2hem.CV1.RS.BetasTrain,2); % n_stim, n_fold
PPtrain(:,2,2) = mean(PP2hem.CV2.RS.BetasTrain,2); 
PPtrain(:,2,3) = mean(PP2hem.CV3.RS.BetasTrain,2); 
PPtrain(:,2,4) = mean(PP2hem.CV4.RS.BetasTrain,2);

%
PTtest(:,2,1) = mean(PT2hem.CV1.RS.BetasTest,2); % n_stim, n_fold
PTtest(:,2,2) = mean(PT2hem.CV2.RS.BetasTest,2); 
PTtest(:,2,3) = mean(PT2hem.CV3.RS.BetasTest,2); 
PTtest(:,2,4) = mean(PT2hem.CV4.RS.BetasTest,2); 

PTtrain(:,2,1) = mean(PT2hem.CV1.RS.BetasTrain,2); % n_stim, n_fold
PTtrain(:,2,2) = mean(PT2hem.CV2.RS.BetasTrain,2); 
PTtrain(:,2,3) = mean(PT2hem.CV3.RS.BetasTrain,2); 
PTtrain(:,2,4) = mean(PT2hem.CV4.RS.BetasTrain,2); 

%
mSTGtest(:,2,1) = mean(mSTG2hem.CV1.RS.BetasTest,2); % n_stim, n_fold
mSTGtest(:,2,2) = mean(mSTG2hem.CV2.RS.BetasTest,2); 
mSTGtest(:,2,3) = mean(mSTG2hem.CV3.RS.BetasTest,2); 
mSTGtest(:,2,4) = mean(mSTG2hem.CV4.RS.BetasTest,2); 

mSTGtrain(:,2,1) = mean(mSTG2hem.CV1.RS.BetasTrain,2); % n_stim, n_fold
mSTGtrain(:,2,2) = mean(mSTG2hem.CV2.RS.BetasTrain,2); 
mSTGtrain(:,2,3) = mean(mSTG2hem.CV3.RS.BetasTrain,2); 
mSTGtrain(:,2,4) = mean(mSTG2hem.CV4.RS.BetasTrain,2); 

%
aSTGtest(:,2,1) = mean(aSTG2hem.CV1.RS.BetasTest,2); % n_stim, n_fold
aSTGtest(:,2,2) = mean(aSTG2hem.CV2.RS.BetasTest,2); 
aSTGtest(:,2,3) = mean(aSTG2hem.CV3.RS.BetasTest,2); 
aSTGtest(:,2,4) = mean(aSTG2hem.CV4.RS.BetasTest,2); 

aSTGtrain(:,2,1) = mean(aSTG2hem.CV1.RS.BetasTrain,2); % n_stim, n_fold
aSTGtrain(:,2,2) = mean(aSTG2hem.CV2.RS.BetasTrain,2); 
aSTGtrain(:,2,3) = mean(aSTG2hem.CV3.RS.BetasTrain,2); 
aSTGtrain(:,2,4) = mean(aSTG2hem.CV4.RS.BetasTrain,2); 

%
pSTGtest(:,2,1) = mean(pSTG2hem.CV1.RS.BetasTest,2); % n_stim, n_fold
pSTGtest(:,2,2) = mean(pSTG2hem.CV2.RS.BetasTest,2); 
pSTGtest(:,2,3) = mean(pSTG2hem.CV3.RS.BetasTest,2); 
pSTGtest(:,2,4) = mean(pSTG2hem.CV4.RS.BetasTest,2); 

pSTGtrain(:,2,1) = mean(pSTG2hem.CV1.RS.BetasTrain,2); % n_stim, n_fold
pSTGtrain(:,2,2) = mean(pSTG2hem.CV2.RS.BetasTrain,2); 
pSTGtrain(:,2,3) = mean(pSTG2hem.CV3.RS.BetasTrain,2); 
pSTGtrain(:,2,4) = mean(pSTG2hem.CV4.RS.BetasTrain,2); 

%%
HGtest(:,3,1) = mean(HG2hem.CV1.RS1.BetasTest,2); % n_stim, n_fold
HGtest(:,3,2) = mean(HG2hem.CV2.RS1.BetasTest,2); 
HGtest(:,3,3) = mean(HG2hem.CV3.RS1.BetasTest,2); 
HGtest(:,3,4) = mean(HG2hem.CV4.RS1.BetasTest,2); 

HGtrain(:,3,1) = mean(HG2hem.CV1.RS1.BetasTrain,2); % n_stim, n_fold
HGtrain(:,3,2) = mean(HG2hem.CV2.RS1.BetasTrain,2); 
HGtrain(:,3,3) = mean(HG2hem.CV3.RS1.BetasTrain,2); 
HGtrain(:,3,4) = mean(HG2hem.CV4.RS1.BetasTrain,2); 

%
PPtest(:,3,1) = mean(PP2hem.CV1.RS1.BetasTest,2); % n_stim, n_fold
PPtest(:,3,2) = mean(PP2hem.CV2.RS1.BetasTest,2); 
PPtest(:,3,3) = mean(PP2hem.CV3.RS1.BetasTest,2); 
PPtest(:,3,4) = mean(PP2hem.CV4.RS1.BetasTest,2); 

PPtrain(:,3,1) = mean(PP2hem.CV1.RS1.BetasTrain,2); % n_stim, n_fold
PPtrain(:,3,2) = mean(PP2hem.CV2.RS1.BetasTrain,2); 
PPtrain(:,3,3) = mean(PP2hem.CV3.RS1.BetasTrain,2); 
PPtrain(:,3,4) = mean(PP2hem.CV4.RS1.BetasTrain,2);

%
PTtest(:,3,1) = mean(PT2hem.CV1.RS1.BetasTest,2); % n_stim, n_fold
PTtest(:,3,2) = mean(PT2hem.CV2.RS1.BetasTest,2); 
PTtest(:,3,3) = mean(PT2hem.CV3.RS1.BetasTest,2); 
PTtest(:,3,4) = mean(PT2hem.CV4.RS1.BetasTest,2); 

PTtrain(:,3,1) = mean(PT2hem.CV1.RS1.BetasTrain,2); % n_stim, n_fold
PTtrain(:,3,2) = mean(PT2hem.CV2.RS1.BetasTrain,2); 
PTtrain(:,3,3) = mean(PT2hem.CV3.RS1.BetasTrain,2); 
PTtrain(:,3,4) = mean(PT2hem.CV4.RS1.BetasTrain,2); 

%
mSTGtest(:,3,1) = mean(mSTG2hem.CV1.RS1.BetasTest,2); % n_stim, n_fold
mSTGtest(:,3,2) = mean(mSTG2hem.CV2.RS1.BetasTest,2); 
mSTGtest(:,3,3) = mean(mSTG2hem.CV3.RS1.BetasTest,2); 
mSTGtest(:,3,4) = mean(mSTG2hem.CV4.RS1.BetasTest,2); 

mSTGtrain(:,3,1) = mean(mSTG2hem.CV1.RS1.BetasTrain,2); % n_stim, n_fold
mSTGtrain(:,3,2) = mean(mSTG2hem.CV2.RS1.BetasTrain,2); 
mSTGtrain(:,3,3) = mean(mSTG2hem.CV3.RS1.BetasTrain,2); 
mSTGtrain(:,3,4) = mean(mSTG2hem.CV4.RS1.BetasTrain,2); 

%
aSTGtest(:,3,1) = mean(aSTG2hem.CV1.RS1.BetasTest,2); % n_stim, n_fold
aSTGtest(:,3,2) = mean(aSTG2hem.CV2.RS1.BetasTest,2); 
aSTGtest(:,3,3) = mean(aSTG2hem.CV3.RS1.BetasTest,2); 
aSTGtest(:,3,4) = mean(aSTG2hem.CV4.RS1.BetasTest,2); 

aSTGtrain(:,3,1) = mean(aSTG2hem.CV1.RS1.BetasTrain,2); % n_stim, n_fold
aSTGtrain(:,3,2) = mean(aSTG2hem.CV2.RS1.BetasTrain,2); 
aSTGtrain(:,3,3) = mean(aSTG2hem.CV3.RS1.BetasTrain,2); 
aSTGtrain(:,3,4) = mean(aSTG2hem.CV4.RS1.BetasTrain,2); 

%
pSTGtest(:,3,1) = mean(pSTG2hem.CV1.RS1.BetasTest,2); % n_stim, n_fold
pSTGtest(:,3,2) = mean(pSTG2hem.CV2.RS1.BetasTest,2); 
pSTGtest(:,3,3) = mean(pSTG2hem.CV3.RS1.BetasTest,2); 
pSTGtest(:,3,4) = mean(pSTG2hem.CV4.RS1.BetasTest,2); 

pSTGtrain(:,3,1) = mean(pSTG2hem.CV1.RS1.BetasTrain,2); % n_stim, n_fold
pSTGtrain(:,3,2) = mean(pSTG2hem.CV2.RS1.BetasTrain,2); 
pSTGtrain(:,3,3) = mean(pSTG2hem.CV3.RS1.BetasTrain,2); 
pSTGtrain(:,3,4) = mean(pSTG2hem.CV4.RS1.BetasTrain,2);
%%
HGtest(:,4,1) = mean(HG2hem.CV1.RS2.BetasTest,2); % n_stim, n_fold
HGtest(:,4,2) = mean(HG2hem.CV2.RS2.BetasTest,2); 
HGtest(:,4,3) = mean(HG2hem.CV3.RS2.BetasTest,2); 
HGtest(:,4,4) = mean(HG2hem.CV4.RS2.BetasTest,2); 

HGtrain(:,4,1) = mean(HG2hem.CV1.RS2.BetasTrain,2); % n_stim, n_fold
HGtrain(:,4,2) = mean(HG2hem.CV2.RS2.BetasTrain,2); 
HGtrain(:,4,3) = mean(HG2hem.CV3.RS2.BetasTrain,2); 
HGtrain(:,4,4) = mean(HG2hem.CV4.RS2.BetasTrain,2); 

%
PPtest(:,4,1) = mean(PP2hem.CV1.RS2.BetasTest,2); % n_stim, n_fold
PPtest(:,4,2) = mean(PP2hem.CV2.RS2.BetasTest,2); 
PPtest(:,4,3) = mean(PP2hem.CV3.RS2.BetasTest,2); 
PPtest(:,4,4) = mean(PP2hem.CV4.RS2.BetasTest,2); 

PPtrain(:,4,1) = mean(PP2hem.CV1.RS2.BetasTrain,2); % n_stim, n_fold
PPtrain(:,4,2) = mean(PP2hem.CV2.RS2.BetasTrain,2); 
PPtrain(:,4,3) = mean(PP2hem.CV3.RS2.BetasTrain,2); 
PPtrain(:,4,4) = mean(PP2hem.CV4.RS2.BetasTrain,2);

%
PTtest(:,4,1) = mean(PT2hem.CV1.RS2.BetasTest,2); % n_stim, n_fold
PTtest(:,4,2) = mean(PT2hem.CV2.RS2.BetasTest,2); 
PTtest(:,4,3) = mean(PT2hem.CV3.RS2.BetasTest,2); 
PTtest(:,4,4) = mean(PT2hem.CV4.RS2.BetasTest,2); 

PTtrain(:,4,1) = mean(PT2hem.CV1.RS2.BetasTrain,2); % n_stim, n_fold
PTtrain(:,4,2) = mean(PT2hem.CV2.RS2.BetasTrain,2); 
PTtrain(:,4,3) = mean(PT2hem.CV3.RS2.BetasTrain,2); 
PTtrain(:,4,4) = mean(PT2hem.CV4.RS2.BetasTrain,2); 

%
mSTGtest(:,4,1) = mean(mSTG2hem.CV1.RS2.BetasTest,2); % n_stim, n_fold
mSTGtest(:,4,2) = mean(mSTG2hem.CV2.RS2.BetasTest,2); 
mSTGtest(:,4,3) = mean(mSTG2hem.CV3.RS2.BetasTest,2); 
mSTGtest(:,4,4) = mean(mSTG2hem.CV4.RS2.BetasTest,2); 

mSTGtrain(:,4,1) = mean(mSTG2hem.CV1.RS2.BetasTrain,2); % n_stim, n_fold
mSTGtrain(:,4,2) = mean(mSTG2hem.CV2.RS2.BetasTrain,2); 
mSTGtrain(:,4,3) = mean(mSTG2hem.CV3.RS2.BetasTrain,2); 
mSTGtrain(:,4,4) = mean(mSTG2hem.CV4.RS2.BetasTrain,2); 

%
aSTGtest(:,4,1) = mean(aSTG2hem.CV1.RS2.BetasTest,2); % n_stim, n_fold
aSTGtest(:,4,2) = mean(aSTG2hem.CV2.RS2.BetasTest,2); 
aSTGtest(:,4,3) = mean(aSTG2hem.CV3.RS2.BetasTest,2); 
aSTGtest(:,4,4) = mean(aSTG2hem.CV4.RS2.BetasTest,2); 

aSTGtrain(:,4,1) = mean(aSTG2hem.CV1.RS2.BetasTrain,2); % n_stim, n_fold
aSTGtrain(:,4,2) = mean(aSTG2hem.CV2.RS2.BetasTrain,2); 
aSTGtrain(:,4,3) = mean(aSTG2hem.CV3.RS2.BetasTrain,2); 
aSTGtrain(:,4,4) = mean(aSTG2hem.CV4.RS2.BetasTrain,2); 

%
pSTGtest(:,4,1) = mean(pSTG2hem.CV1.RS2.BetasTest,2); % n_stim, n_fold
pSTGtest(:,4,2) = mean(pSTG2hem.CV2.RS2.BetasTest,2); 
pSTGtest(:,4,3) = mean(pSTG2hem.CV3.RS2.BetasTest,2); 
pSTGtest(:,4,4) = mean(pSTG2hem.CV4.RS2.BetasTest,2); 

pSTGtrain(:,4,1) = mean(pSTG2hem.CV1.RS2.BetasTrain,2); % n_stim, n_fold
pSTGtrain(:,4,2) = mean(pSTG2hem.CV2.RS2.BetasTrain,2); 
pSTGtrain(:,4,3) = mean(pSTG2hem.CV3.RS2.BetasTrain,2); 
pSTGtrain(:,4,4) = mean(pSTG2hem.CV4.RS2.BetasTrain,2);
%%
HGtest(:,5,1) = mean(HG2hem.CV1.RS3.BetasTest,2); % n_stim, n_fold
HGtest(:,5,2) = mean(HG2hem.CV2.RS3.BetasTest,2); 
HGtest(:,5,3) = mean(HG2hem.CV3.RS3.BetasTest,2); 
HGtest(:,5,4) = mean(HG2hem.CV4.RS3.BetasTest,2); 

HGtrain(:,5,1) = mean(HG2hem.CV1.RS3.BetasTrain,2); % n_stim, n_fold
HGtrain(:,5,2) = mean(HG2hem.CV2.RS3.BetasTrain,2); 
HGtrain(:,5,3) = mean(HG2hem.CV3.RS3.BetasTrain,2); 
HGtrain(:,5,4) = mean(HG2hem.CV4.RS3.BetasTrain,2); 

%
PPtest(:,5,1) = mean(PP2hem.CV1.RS3.BetasTest,2); % n_stim, n_fold
PPtest(:,5,2) = mean(PP2hem.CV2.RS3.BetasTest,2); 
PPtest(:,5,3) = mean(PP2hem.CV3.RS3.BetasTest,2); 
PPtest(:,5,4) = mean(PP2hem.CV4.RS3.BetasTest,2); 

PPtrain(:,5,1) = mean(PP2hem.CV1.RS3.BetasTrain,2); % n_stim, n_fold
PPtrain(:,5,2) = mean(PP2hem.CV2.RS3.BetasTrain,2); 
PPtrain(:,5,3) = mean(PP2hem.CV3.RS3.BetasTrain,2); 
PPtrain(:,5,4) = mean(PP2hem.CV4.RS3.BetasTrain,2);

%
PTtest(:,5,1) = mean(PT2hem.CV1.RS3.BetasTest,2); % n_stim, n_fold
PTtest(:,5,2) = mean(PT2hem.CV2.RS3.BetasTest,2); 
PTtest(:,5,3) = mean(PT2hem.CV3.RS3.BetasTest,2); 
PTtest(:,5,4) = mean(PT2hem.CV4.RS3.BetasTest,2); 

PTtrain(:,5,1) = mean(PT2hem.CV1.RS3.BetasTrain,2); % n_stim, n_fold
PTtrain(:,5,2) = mean(PT2hem.CV2.RS3.BetasTrain,2); 
PTtrain(:,5,3) = mean(PT2hem.CV3.RS3.BetasTrain,2); 
PTtrain(:,5,4) = mean(PT2hem.CV4.RS3.BetasTrain,2); 

%
mSTGtest(:,5,1) = mean(mSTG2hem.CV1.RS3.BetasTest,2); % n_stim, n_fold
mSTGtest(:,5,2) = mean(mSTG2hem.CV2.RS3.BetasTest,2); 
mSTGtest(:,5,3) = mean(mSTG2hem.CV3.RS3.BetasTest,2); 
mSTGtest(:,5,4) = mean(mSTG2hem.CV4.RS3.BetasTest,2); 

mSTGtrain(:,5,1) = mean(mSTG2hem.CV1.RS3.BetasTrain,2); % n_stim, n_fold
mSTGtrain(:,5,2) = mean(mSTG2hem.CV2.RS3.BetasTrain,2); 
mSTGtrain(:,5,3) = mean(mSTG2hem.CV3.RS3.BetasTrain,2); 
mSTGtrain(:,5,4) = mean(mSTG2hem.CV4.RS3.BetasTrain,2); 

%
aSTGtest(:,5,1) = mean(aSTG2hem.CV1.RS3.BetasTest,2); % n_stim, n_fold
aSTGtest(:,5,2) = mean(aSTG2hem.CV2.RS3.BetasTest,2); 
aSTGtest(:,5,3) = mean(aSTG2hem.CV3.RS3.BetasTest,2); 
aSTGtest(:,5,4) = mean(aSTG2hem.CV4.RS3.BetasTest,2); 

aSTGtrain(:,5,1) = mean(aSTG2hem.CV1.RS3.BetasTrain,2); % n_stim, n_fold
aSTGtrain(:,5,2) = mean(aSTG2hem.CV2.RS3.BetasTrain,2); 
aSTGtrain(:,5,3) = mean(aSTG2hem.CV3.RS3.BetasTrain,2); 
aSTGtrain(:,5,4) = mean(aSTG2hem.CV4.RS3.BetasTrain,2); 

%
pSTGtest(:,5,1) = mean(pSTG2hem.CV1.RS3.BetasTest,2); % n_stim, n_fold
pSTGtest(:,5,2) = mean(pSTG2hem.CV2.RS3.BetasTest,2); 
pSTGtest(:,5,3) = mean(pSTG2hem.CV3.RS3.BetasTest,2); 
pSTGtest(:,5,4) = mean(pSTG2hem.CV4.RS3.BetasTest,2); 

pSTGtrain(:,5,1) = mean(pSTG2hem.CV1.RS3.BetasTrain,2); % n_stim, n_fold
pSTGtrain(:,5,2) = mean(pSTG2hem.CV2.RS3.BetasTrain,2); 
pSTGtrain(:,5,3) = mean(pSTG2hem.CV3.RS3.BetasTrain,2); 
pSTGtrain(:,5,4) = mean(pSTG2hem.CV4.RS3.BetasTrain,2);

%%
clear dat_train dat_test
dat_train = cat(4, HGtrain,PTtrain,PPtrain,mSTGtrain ,pSTGtrain, aSTGtrain);
dat_test = cat(4, HGtest,PTtest,PPtest,mSTGtest ,pSTGtest, aSTGtest);
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