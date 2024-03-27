clear
clc
wd = 'C:\Users\Martin\Desktop\EMGVB_REV';
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
lm.ll_test          = glm_ll(data.test,lm.beta);

lm.perf             = [nan,lm.ll_train,lm.perf_train, nan,lm.ll_test,lm.perf_test];

setting.Prior.Mu    = Mdl.PriorVal(1);
setting.Prior.Sig   = Mdl.PriorVal(2);


%%

opt.lr              = 0.01;
opt.MaxIter         = 1000;
opt.MaxPatience     = 2000;
opt.StepAdaptive    = 2000;
opt.GradientMax     = 3000;
opt.NumSample       = 50;
opt.SigInitScale    = 0.05;
opt.GradClipInit    = 1000;

%% Train Test

setting.isDiag      = 0;
setting.useHfunc    = 0;
clc

rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);

rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);


rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);


plot(pEMGVB.Post.LB_smooth)
hold on
plot(pMGVB.Post.LB_smooth,'--')
plot(pQBVI.Post.LB_smooth,':r')
hold off

%%

setting.isDiag      = 0;
setting.useHfunc    = 1;

rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);

rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);

rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);



%%

setting.useHfunc    = 0;
setting.isDiag      = 1;

clc
rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);

rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);

rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);

%%

setting.isDiag      = 1;
setting.useHfunc    = 1;


rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);

rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);

rng(seed)
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
    'SaveParams',true,...
    'Verbose',2,...
    'LBPlot',false);

%% MCMC Sampler
clc
Run_MCMC = 1;

if Run_MCMC == 1
    rng(seed)
    mcmc.factor = 1;
    mcmc.N_MC_samples = 200000;
    Post_MCMC = MCMC(Mdl,data.train,'NumMCMC',mcmc.N_MC_samples*mcmc.factor,'ParamsInit',lm.beta,'Verbose',1000);

    [mcmc.mean,mcmc.std,mcmc.chain] = Post_MCMC.getParamsMean('BurnInRate',0.2);
    mcmc.par        = mcmc.mean';
    mcmc.train_perf = perf_measures(data.train,mcmc.par);
    mcmc.test_perf  = perf_measures(data.test,mcmc.par);
    mcmc.train_ll   = glm_ll(data.train,mcmc.par);
    mcmc.test_ll    = glm_ll(data.test,mcmc.par);
    mcmc.perf       = [nan,mcmc.train_ll,mcmc.train_perf, nan,mcmc.test_ll,mcmc.test_perf];
end

%%

f = @(x) x.Post.par;
tab_par = [f(pEMGVB);f(pMGVB);f(pQBVI); f(pEMGVB_h);f(pMGVB_h);f(pQBVI_h);...
    f(pEMGVB_d);f(pMGVB_d);f(pQBVI_d); f(pEMGVB_h_d);f(pMGVB_h_d);f(pQBVI_h_d);...
    mcmc.par';lm.beta'];

f = @(x) x.Post.perf_short;
tab_perf = [f(pEMGVB);f(pMGVB);f(pQBVI); f(pEMGVB_h);f(pMGVB_h);f(pQBVI_h);...
    f(pEMGVB_d);f(pMGVB_d);f(pQBVI_d); f(pEMGVB_h_d);f(pMGVB_h_d);f(pQBVI_h_d);...
    mcmc.perf;lm.perf];


sig.glm     = triu(lm.glm.covb - diag(diag(lm.glm.covb)));
sig.qbvi    = tril(pQBVI.Post.Sig - diag(pQBVI.Post.Sig2));
sig.emgvb   = tril(pEMGVB.Post.Sig - diag(pEMGVB.Post.Sig2));
sig.mgvb    = triu(pMGVB.Post.Sig - diag(pMGVB.Post.Sig2));
sig.mcmc    = tril(cov(mcmc.chain)-diag(diag(cov(mcmc.chain))));

sig.emgvb_mgvb  = sig.emgvb + sig.mgvb;
sig.mcmc_glm    = sig.mcmc + sig.glm;

f = @(x) x.Post.Sig2';
tab_var = [f(pEMGVB);f(pMGVB);f(pQBVI); f(pEMGVB_h);f(pMGVB_h);f(pQBVI_h);...
    f(pEMGVB_d);f(pMGVB_d);f(pQBVI_d); f(pEMGVB_h_d);f(pMGVB_h_d);f(pQBVI_h_d);...
    mcmc.std.^2;lm.glm.se'.^2];


f = @(x) x.Post.iter.logq(x.Post.iter_best);
tab_logq = [f(pEMGVB);f(pMGVB);f(pQBVI); f(pEMGVB_h);f(pMGVB_h);f(pQBVI_h);...
    f(pEMGVB_d);f(pMGVB_d);f(pQBVI_d); f(pEMGVB_h_d);f(pMGVB_h_d);f(pQBVI_h_d);...
    nan;nan];

%%

Run = 1;

setting.isDiag      = 0;
setting.useHfunc    = 0;

if Run
    n_list  = [10,20,30,50,75,100,150,200,300];    
    N       = cell(numel(n_list),3);

    for i = 1:numel(n_list)

        fprintf('N: %i.\n',i)
        rng(seed)
        N{i,1} = n_list(i);
        tic
        N{i,2} = EMGVB_labour(@h_func_labour,data,...
            'NumParams',8,...
            'Setting',setting,...
            'LearningRate',opt.lr,...
            'NumSample',n_list(i),...
            'MaxPatience',opt.MaxPatience,...
            'MaxIter',opt.MaxIter,...
            'GradWeight',0.4,...
            'WindowSize',30,...
            'GradClipInit',opt.GradClipInit,...
            'SigInitScale',opt.SigInitScale,...
            'StepAdaptive',opt.StepAdaptive,...
            'GradientMax',opt.GradientMax,...
            'SaveParams',true,...
            'Verbose',2,...
            'LBPlot',false);

        N{i,3} = toc;
    end

    f = @(i) [N{i,2}.Post.par,N{i,2}.Post.LB0,N{i,2}.Post.perf_short];
    tab_n = [n_list',cell2mat(arrayfun(@(i) f(i),1:numel(n_list),'uni',0)'),[N{:,3}]'];

end

[pEMGVB.Post.par;N{n_list==75,2}.Post.par]
[pEMGVB.Post.perf_short;N{n_list==75,2}.Post.perf_short]


%%


save('Labour\Results\Results.mat',...
    'data', 'setting', 'lm', 'Mdl', 'n_features', 'opt',...
    'pMGVB', 'pEMGVB', 'pQBVI',...
    'pMGVB_h', 'pEMGVB_h', 'pQBVI_h',...
    'pMGVB_d', 'pEMGVB_d', 'pQBVI_d',...
    'pMGVB_h_d', 'pEMGVB_h_d', 'pQBVI_h_d', 'mcmc','sig',...
    'tab_par','tab_perf','tab_var','tab_logq','seed',...
    'N','tab_n')




writematrix(tab_par,'Labour\Results\Labour_tables.xls','Sheet','par')
writematrix(tab_perf,'Labour\Results\Labour_tables.xls','Sheet','perf')
writematrix(tab_var,'Labour\Results\Labour_tables.xls','Sheet','var')
writematrix(sig.emgvb_mgvb,'Labour\Results\Labour_tables.xls','Sheet','cov_emgvb_mgvb')
writematrix(sig.mcmc_glm,'Labour\Results\Labour_tables.xls','Sheet','cov_mcmc_glm')
writematrix(tab_logq,'Labour\Results\Labour_tables.xls','Sheet','tab_logq')
writematrix(tab_n,'Labour\Results\Labour_tables.xls','Sheet','tab_n')
