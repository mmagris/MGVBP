clear
clc
wd = 'C:\Users\Martin\Desktop\EMGVB_REV';
cd(wd)
addpath(genpath('VBLab'))
addpath('Utils')
addpath(genpath('Istanbul'))
addpath(genpath('VBLab'))
rmpath(genpath('VBLab\VB\MGVB'))

%%

z = readtable("data_akbilgic.xlsx");
z = z(:,3:end);
z.const = ones(size(z,1),1);

z = z(:,{'ISE_1','const','SP','NIKKEI','BOVESPA','DAX','FTSE','EU','EM'});
data = z{:,:};

n               = floor(size(data,1)*0.80);
Data.train      = data(1:n,:);
Data.test       = data(n+1:end,:);
Data.n.train    = n;
tab.train       = z(1:n,:);
tab.test        = z(n+1:end,:);

lm = fitlm(tab.train,'ISE_1~const+SP+NIKKEI+BOVESPA+DAX+FTSE+EU+EM','Intercept',0);

%%
clc

opt.lr              = 0.1;  
opt.NumSample       = 50;
opt.MaxPatience     = 500;
opt.MaxIter         = 1200;
opt.StepAdaptive    = 1000;
opt.GradientMax     = 50000;
opt.GradClipInit    = 1000;
opt.SigInitScale    = 0.01;
opt.Verbose         = 2;
opt.MeanInit        = zeros(9,1);
seed = 2022;

setting.Prior.Mu        = [0,0,0,0,0,0,0,0,0];
setting.Prior.Sig       = 5;
setting.Block.blks      = [9];
setting.useHfunc        = 1;


clc

rng(seed)
bEMGVB = EMGVBb(@h_func_b,Data.train,...
    'NumParams',9,...    
    'Setting',setting,...
    'LearningRate',opt.lr,...
    'NumSample',opt.NumSample,...
    'MaxPatience',opt.MaxPatience,...
    'MaxIter',opt.MaxIter,...
    'GradWeight',0.4,...
    'WindowSize',30,...
    'SigInitScale',opt.SigInitScale,...
    'StepAdaptive',opt.StepAdaptive,...
    'GradientMax',opt.GradientMax,...
    'GradClipInit',opt.GradClipInit,...
    'SaveParams',true,...
    'Verbose',opt.Verbose,...  
    'LBPlot',true);

m.full.sig  = exp(bEMGVB.Post.mu(end)+bEMGVB.Post.Sig2(end)/2);
m.full.perf = lm_perf(Data,bEMGVB.Post.mu(1:end-1),m.full.sig);
m.full.mdl  = bEMGVB;


[bEMGVB.Post.mu,[lm.Coefficients.Estimate;lm.RMSE]]
[m.full.mdl.Post.Sig2,[diag(lm.CoefficientCovariance);nan]]*1000

%%%

setting.Block.blks      = [1,1,1,1,1,1,1,1,1];

clc

rng(seed)
bEMGVB = EMGVBb(@h_func_b,Data.train,...
    'NumParams',9,...
    'MeanInit',opt.MeanInit,...
    'Setting',setting,...
    'LearningRate',opt.lr,...
    'NumSample',opt.NumSample,...
    'MaxPatience',opt.MaxPatience,...
    'MaxIter',opt.MaxIter,...
    'GradWeight',0.4,...
    'WindowSize',30,...
    'SigInitScale',opt.SigInitScale,...
    'StepAdaptive',opt.StepAdaptive,...
    'GradientMax',opt.GradientMax,...
    'GradClipInit',opt.GradClipInit,...
    'SaveParams',true,...
    'Verbose',opt.Verbose,...  
    'LBPlot',true);


m.diag.sig  = exp(bEMGVB.Post.mu(end)+bEMGVB.Post.Sig2(end)/2);
m.diag.perf = lm_perf(Data,bEMGVB.Post.mu(1:end-1),m.diag.sig);
m.diag.mdl  = bEMGVB;
[m.diag.mdl.Post.Sig2,[diag(lm.CoefficientCovariance);nan]]*1000

%%%

setting.Block.blks      = [8,1];

clc

rng(seed)
bEMGVB = EMGVBb(@h_func_b,Data.train,...
    'NumParams',9,...
    'MeanInit',opt.MeanInit,...
    'Setting',setting,...
    'LearningRate',opt.lr,...
    'NumSample',opt.NumSample,...
    'MaxPatience',opt.MaxPatience,...
    'MaxIter',opt.MaxIter,...
    'GradWeight',0.4,...
    'WindowSize',30,...
    'SigInitScale',opt.SigInitScale,...
    'StepAdaptive',opt.StepAdaptive,...
    'GradientMax',opt.GradientMax,...
    'GradClipInit',opt.GradClipInit,...
    'SaveParams',true,...
    'Verbose',opt.Verbose,...  
    'LBPlot',true);

m.b1.sig  = exp(bEMGVB.Post.mu(end)+bEMGVB.Post.Sig2(end)/2);
m.b1.perf = lm_perf(Data,bEMGVB.Post.mu(1:end-1),m.b1.sig);
m.b1.mdl  = bEMGVB;
[m.b1.mdl.Post.Sig2,[diag(lm.CoefficientCovariance);nan]]*1000

%%%

seed = 2022;

setting.Block.blks      = [1,3,2,2,1];

clc

rng(seed)
bEMGVB = EMGVBb(@h_func_b,Data.train,...
    'NumParams',9,...
    'MeanInit',opt.MeanInit,...
    'Setting',setting,...
    'LearningRate',opt.lr,...
    'NumSample',opt.NumSample,...
    'MaxPatience',opt.MaxPatience,...
    'MaxIter',opt.MaxIter,...
    'GradWeight',0.4,...
    'WindowSize',30,...
    'SigInitScale',opt.SigInitScale,...
    'StepAdaptive',opt.StepAdaptive,...
    'GradientMax',opt.GradientMax,...
    'GradClipInit',opt.GradClipInit,...
    'SaveParams',true,...
    'Verbose',opt.Verbose,...  
    'LBPlot',true);

m.b2.sig  = exp(bEMGVB.Post.mu(end)+bEMGVB.Post.Sig2(end)/2);
m.b2.perf = lm_perf(Data,bEMGVB.Post.mu(1:end-1),m.b2.sig);
m.b2.mdl  = bEMGVB;
[m.b2.mdl.Post.Sig2,[diag(lm.CoefficientCovariance);nan]]*1000

save('Istanbul\Save\Istanbul.mat','m','opt','setting','Data','tab','lm')



%%

load('Istanbul.mat')

lw = 1.4;
flb = @(x) m.(x).mdl.Post.LB_smooth;
xp = 31:opt.MaxIter;

plot(xp,flb('full'),'LineWidth',lw)
hold on
plot(xp,flb('diag'),'LineWidth',lw)
plot(xp,flb('b1'),'LineWidth',lw)
plot(xp,flb('b2'),'LineWidth',lw)
hold off

% leg = legend({'Full (case i)','Diagonal (case ii) ','Block 1 (case iii)','Block 2 (case iv)'},'Location','southeast','Interpreter','latex')
leg = legend({'Full','Diagonal','Block 1','Block 2'},'Location','southeast','Interpreter','latex')
% grid
xlim([0,opt.MaxIter])
ylim([600,1200])
xlabel('Iteration','Interpreter','latex')
title('Lower bound','Interpreter','latex')


set_fig(1600,400,300,180)
savePDF('Istanbul\Plots\','Istanbul_LB')

%%

fmu = @(x) [m.(x).mdl.Post.mu(1:end-1);m.(x).sig];
mat_mu = [fmu('full'),fmu('diag'),fmu('b1'),fmu('b2'),[lm.Coefficients.Estimate;lm.RMSE]]'

%%

clc

flbmax = @(x) array2table(m.(x).mdl.Post.LB_max,'VariableNames',{'LB'});
fperf  = @(x) struct2table(m.(x).perf);
fstat  = @(x) [flbmax(x),fperf(x)];

lm_stat = [array2table(nan,'VariableNames',{'LB'}),struct2table(lm_perf(Data,lm.Coefficients.Estimate,lm.RMSE))];
mat_stat = [fstat('full');fstat('diag');fstat('b1');fstat('b2');lm_stat];

mat_stat.mse_train = mat_stat.mse_train*10^5;
mat_stat.mse_test = mat_stat.mse_test*10^5;
mat_stat = mat_stat(:,[1,2,3,5,4]);

%%

fcov = @(x) m.(x).mdl.Post.Sig;

c_full = fcov('full');
c_diag = diag(fcov('diag'));
c_b1 = fcov('b1');
c_b2 = fcov('b2');

c_ml = lm.CoefficientCovariance;
c_ml(end+1,:) = nan(1,8);
c_ml(:,end+1) = nan(9,1);

se_mat = [m.full.mdl.Post.Sig2,...
            m.diag.mdl.Post.Sig2,...
            m.b1.mdl.Post.Sig2,...
            m.b2.mdl.Post.Sig2,...
            diag(c_ml)];

se_mat = sqrt(se_mat)'*100;


c_b1_b2 = triu(c_b1)-diag(diag(c_b1)) + tril(c_b2) - diag(diag(c_b2));
c_b1_b2 = c_b1_b2*1000;

c_fu_ml = triu(c_full)-diag(diag(c_full)) + tril(c_ml) - diag(diag(c_ml));
c_fu_ml = c_fu_ml*1000;

%%

save_name = 'Istanbul\Save\Istanbul_tables.xls';

writematrix(se_mat,save_name,'Sheet','se')
writematrix(c_b1_b2,save_name,'Sheet','c_b1_b2')
writematrix(c_fu_ml,save_name,'Sheet','c_fu_ml')
writetable(mat_stat,save_name,'Sheet','mat_stat')
writematrix(mat_mu,save_name,'Sheet','mat_mu')
