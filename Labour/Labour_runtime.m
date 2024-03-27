clear
clc
wd = 'C:\Users\Martin\Desktop\EMGVB_ICLM\Code\EMGVB';
cd(wd)
addpath(genpath('VBLab'))
addpath(genpath('Labour'))

seed = 2022;
rng(seed)


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


%%

opt.lr              = 0.01;
opt.MaxIter         = 1000;
opt.MaxPatience     = 2000;
opt.StepAdaptive    = 2000;
opt.GradientMax     = 3000;
opt.NumSample       = 75;
opt.SigInitScale    = 0.05;
opt.GradClipInit    = 1000;

max_i = 20;

fsavename = @(setting) ['Labour\Runtime\RuntimeLabour_isDiag' num2str(setting.isDiag) '_hFunc' num2str(setting.useHfunc) '.mat'];


%% Train Test


t = zeros(3,max_i);

setting.isDiag      = 0;
setting.useHfunc    = 0;
clc

for i = 1:max_i
    rng(seed)
    tic
    pMGVB = MGVB_labour(@h_func_labour,data,...
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
    t(1,i) = toc;

    rng(seed)
    tic
    pEMGVB = EMGVB_labour(@h_func_labour,data,...
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
    t(2,i) = toc;

    rng(seed)
    tic
    pQBVI = QBVI_labour(@h_func_labour,data,...
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
    t(3,i) = toc;
    fprintf('Step 1, isDiag: %i, hFunc: %i, %i/%i.\n',setting.isDiag,setting.useHfunc,i,max_i)
end

time =  mktab(t,setting);
save(fsavename(setting),'time','max_i','opt','setting')


%%

setting.isDiag      = 0;
setting.useHfunc    = 1;

for i = 1:max_i
    rng(seed)
    tic
    pMGVB_h = MGVB_labour(@h_func_labour,data,...
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
    t(1,i) = toc;

    rng(seed)
    tic
    pEMGVB_h = EMGVB_labour(@h_func_labour,data,...
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
    t(2,i) = toc;

    rng(seed)
    tic
    pQBVI_h = QBVI_labour(@h_func_labour,data,...
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
    t(3,i) = toc;
        fprintf('Step 2, isDiag: %i, hFunc: %i, %i/%i.\n',setting.isDiag,setting.useHfunc,i,max_i)
end


time =  mktab(t,setting);
save(fsavename(setting),'time','max_i','opt','setting')


%%

setting.isDiag      = 1;
setting.useHfunc    = 0;


clc

for i = 1:max_i
    rng(seed)
    tic
    pMGVB_d = MGVB_labour(@h_func_labour,data,...
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
    t(1,i) = toc;

    rng(seed)
    tic
    pEMGVB_d = EMGVB_labour(@h_func_labour,data,...
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
    t(2,i) = toc;

    rng(seed)
    tic
    pQBVI_d = QBVI_labour(@h_func_labour,data,...
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
    t(3,i) = toc;
    fprintf('Step 3, isDiag: %i, hFunc: %i, %i/%i.\n',setting.isDiag,setting.useHfunc,i,max_i)
end


time =  mktab(t,setting);
save(fsavename(setting),'time','max_i','opt','setting')

%%

setting.isDiag      = 1;
setting.useHfunc    = 1;

for i = 1:max_i
    rng(seed)
    tic
    pMGVB_h_d = MGVB_labour(@h_func_labour,data,...
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
    t(2,i) = toc;

    rng(seed)
    tic
    pEMGVB_h_d = EMGVB_labour(@h_func_labour,data,...
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
    t(1,i) = toc;

    rng(seed)
    tic
    pQBVI_h_d = QBVI_labour(@h_func_labour,data,...
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
    t(3,i) = toc;
        fprintf('Step 4, isDiag: %i, hFunc: %i, %i/%i.\n',setting.isDiag,setting.useHfunc,i,max_i)
end

runtimeLabour
time =  mktab(t,setting);
% save(fsavename(setting),'time','max_i','opt','setting')

%%

function[time] = mktab(t,setting)
time.t = t;
to = t;
to(isoutlier(t,2)) = nan;
time.to = to;
time.med = median(t,2);
time.mea = mean(t,2);
time.medo = median(to,2,'omitmissing');
time.meao = mean(to,2,'omitmissing');
time.tab = array2table([time.mea,time.med,std(t,[],2)]);
time.tabo = array2table([time.meao,time.medo,std(t,[],2)]);
time.tab.Properties.VariableNames = {'Mea','Med','Std'};
time.tab.Properties.RowNames = {'MGVB','EMGVB','QBVI'};
time.tabo.Properties = time.tab.Properties;
time.isDiag = setting.isDiag;
time.hFunc = setting.useHfunc;
end