clear
clc
wd = 'C:\Users\Martin\Desktop\EMGVB_ICLM\Code\EMGVB';
addpath(genpath('VBLab'))
addpath(genpath('Labour'))

seed = 2022;
rng(seed)

cd = [wd,'\Labour'];

labour = readData('LabourForce','Type','Matrix','Intercept', true);

rng(seed)
indx = randperm(size(labour,1));
labour_perm = labour(indx,:);

indx_train  = floor(size(labour_perm,1)*0.75);
data.train  = labour_perm(1:indx_train,:);
data.test   = labour_perm(indx_train+1:end,:);


n_features = size(labour,2)-1;

% Create a Logistic Regression model object
Mdl = LogisticRegression(n_features,'Prior',{'Normal',[0,5]});


[lm.beta,~,lm.glm]  = glmfit(data.train(:,1:end-1),data.train(:,end),'binomial','constant','off'); % initialise mu
lm.perf_train       = perf_measures(data.train,lm.beta);
lm.perf_test        = perf_measures(data.train,lm.beta);

lm.ll_train         = glm_ll(data.train,lm.beta);
lm.ll_test         = glm_ll(data.test,lm.beta);

lm.perf             = [nan,lm.ll_train,lm.perf_train, nan,lm.ll_test,lm.perf_test];

setting.Prior.Mu    = Mdl.PriorVal(1);
setting.Prior.Sig   = Mdl.PriorVal(2);

path = 'C:\Users\Martin\Desktop\EMGVB_ICLM\Code\EMGVB\Labour\StartingValues';
%%

opt.lr              = 0.01;
opt.MaxIter         = 2500;
opt.MaxPatience     = 2000;
opt.StepAdaptive    = 2000;
opt.GradientMax     = 3000;
opt.NumSample       = 75;
opt.SigInitScale    = 0.05;
opt.GradClipInit    = 1000;

setting.isDiag      = 0;
setting.useHfunc    = 0;
max_i = 20;

%% Random

clc
M = [0,0,0,0,5,5,5,5,-5,-5,-5,-5];
S = [0.1,1,5,10,0.1,1,5,10,0.1,1,5,10];

R  = cell(numel(M),6);
Ro = cell(numel(M),6);
RTMP = cell(numel(M),2);

for j = 1:numel(M)

    tmp = zeros(max_i,5);
    for i = 1:max_i

        
        pEMGVB = EMGVB_labour(@h_func_labour,data,...
            'MeanInit',normrnd(M(j),S(j),8,1),...
            'NumParams',8,...
            'Setting',setting,...
            'LearningRate',opt.lr,...
            'NumSample',opt.NumSample,...
            'MaxPatience',opt.MaxPatience,...
            'MaxIter',opt.MaxIter,...
            'GradWeight',0.4,...
            'WindowSize',30,...
            'GradClipInit',opt.GradClipInit,...
            'SigInitScale',opt.SigInitScale,...
            'StepAdaptive',opt.StepAdaptive,...
            'GradientMax',opt.GradientMax,...
            'SaveParams',false,...
            'Verbose',0,...
            'LBPlot',false);

        tmp(i,:) = [pEMGVB.Post.LB_max, pEMGVB.Post.train.perf_best];

        fprintf('Random. j: %i/%i, i: %i/%i, LB: %.3f.\n',j,numel(M),i,max_i,pEMGVB.Post.LB_max)

    end

    tmpo     = tmp(~isoutlier(tmp(:,1)),:);

    RTMP{j,1} = tmp;   
    RTMP{j,2} = tmpo;   

    R   = mkcell(R,j,M,S,tmp);    
    Ro  = mkcell(Ro,j,M,S,tmpo);

end

tabR    = mktab(R);
tabRo   = mktab(Ro);

save('Labour\StartingValues\StartVals_Rand.mat','setting','opt','R','Ro','tabR','tabRo','RTMP','pEMGVB')

%% Fixed

M = [0,1,-1,5,-5,20,-20];
S = zeros(1,numel(M));

F  = cell(numel(M),6);
Fo = cell(numel(M),6);
FTMP = cell(numel(M),2);

for j = 1:numel(M)
    
    tmp = zeros(max_i,5);
    for i = 1:max_i

        
        pEMGVB = EMGVB_labour(@h_func_labour,data,...
            'MeanInit',M(j) + zeros(8,1),...
            'NumParams',8,...
            'Setting',setting,...
            'LearningRate',opt.lr,...
            'NumSample',opt.NumSample,...
            'MaxPatience',opt.MaxPatience,...
            'MaxIter',opt.MaxIter,...
            'GradWeight',0.4,...
            'WindowSize',30,...
            'GradClipInit',opt.GradClipInit,...
            'SigInitScale',opt.SigInitScale,...
            'StepAdaptive',opt.StepAdaptive,...
            'GradientMax',opt.GradientMax,...
            'SaveParams',false,...
            'Verbose',0,...
            'LBPlot',false);

        tmp(i,:) = [pEMGVB.Post.LB_max, pEMGVB.Post.train.perf_best];

        fprintf('Fixed. j: %i/%i, i: %i/%i, LB: %.3f.\n',j,numel(M),i,max_i,pEMGVB.Post.LB_max)

    end

    tmpo     = tmp(~isoutlier(tmp(:,1)),:);

    FTMP{j,1} = tmp;   
    FTMP{j,2} = tmpo;   

    F   = mkcell(F,j,M,S,tmp);    
    Fo  = mkcell(Fo,j,M,S,tmpo);

end

tabF    = mktab(F);
tabFo   = mktab(Fo);


save('Labour\StartingValues\StartVals_Fixed.mat','setting','opt','F','Fo','tabF','tabFo','FTMP','pEMGVB')

%%
    
% load(fullfile(path,'StartVals_Rand.mat'))
% load(fullfile(path,'StartVals_Fixed.mat'))

writetable(tabRo,fullfile(path,'Labour_StartValues.xlsx'),'Sheet','Rand')
writetable(tabFo,fullfile(path,'Labour_StartValues.xlsx'),'Sheet','Fixed')


%%


function[tabF] = mktab(F)
tabF = array2table([vertcat(F{:,1}),vertcat(F{:,2}),vertcat(F{:,4}),vertcat(F{:,5}),vertcat(F{:,6})]);
tabF.Properties.VariableNames = {'Mu','Sig','LB','A','P','R','F1','medLB','medA','medP','medR','medF1','sLB','sa','sP','sR','sF1'};
end

function[F] = mkcell(F,j,M,S,tmp)
F{j,1} = M(j);
F{j,2} = S(j);
F{j,3} = tmp;
F{j,4} = mean(tmp,1);
F{j,5} = median(tmp,1);
F{j,6} = std(tmp,[],1);
end