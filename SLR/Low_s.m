
wd = 'C:\Users\Martin\Desktop\EMGVB_REV\SLR';
cd(wd)
addpath(genpath('SLR'))


clear
clc
close all

n = 100;
x = linspace(0,5,n)';
sig2 = 1;
y = 2*x + normrnd(0,sqrt(sig2),n,1);

mu0 = 1;
S0  = 1;
ret = @(S,xi) S + xi +0.5*xi*inv(S)*xi;

Sn  = inv(inv(S0) + x'*x/sig2)
Mn  = Sn*(inv(S0)*mu0+x'*y/sig2);
plot(x,y)

kl  = @(mup,muq,Sp,Sq) log(sqrt(Sq)/sqrt(Sp))+(Sp+(mup-muq)^2)/(2*Sq)-1/2;
KL  = @(muq,Sq) kl(Mn,muq,Sn,Sq);

%%

clc
optim = 'emgvbp';

MaxIter = 500;
mu_init = 0;
S_init  = 0.1;

beta = 0.001;

mu = mu_init;
S  = S_init;
Ns = 500;


LB = nan(MaxIter,1);
natgrad_lb_mu = nan(MaxIter,1);
natgrad_lb_S = nan(MaxIter,1);
MS = nan(MaxIter,2);
K = nan(MaxIter,1);


rng(1)

for iter = 1:MaxIter
    grad_lb_mu = nan(1,Ns);
    grad_lb_S = nan(1,Ns);
    lb = nan(1,Ns);

    iS = inv(S);
    for s = 1:Ns
        theta = normrnd(mu,sqrt(S),1,1);

        aux = (y-theta*x).^2;
        log_ll = -n/2*log(2*pi)-n/2*log(sig2)-1/2*sum(aux/sig2);
        log_q  = -1/2*log(2*pi)-1/2*log(S)-1/2*(theta-mu)^2/S;
        log_p  = -1/2*log(2*pi)-1/2*log(S0)-1/2*(theta-mu0)^2/S0;


        grad_log_q_mu = (theta-mu);
        grad_log_q_S  = -1/2*(iS - iS*(theta-mu)*(theta-mu)'*iS);

        h = log_ll + log_p - log_q;

        lb(1,s) = h;
        grad_lb_mu(1,s)  = grad_log_q_mu*h;
        grad_lb_S(1,s)   = grad_log_q_S*h;

    end

    LB(iter) = mean(lb);

    natgrad_lb_mu(iter) = mean(grad_lb_mu);
    natgrad_lb_S(iter)  = 2*S*mean(grad_lb_S)*S;


    beta_use = beta;

    mu = mu +beta_use*natgrad_lb_mu(iter);
    switch optim
        case 'mgvb'
            S  = ret(S,beta_use*natgrad_lb_S(iter));
        case 'emgvb'
            S  = ret(S,beta_use/2*natgrad_lb_S(iter));
        otherwise
            iS  = ret(iS,-beta_use*mean(grad_lb_S));
            S = inv(iS);
    end

    MS(iter,:) = [mu,S];
    k = KL(mu,S);
    K(iter,1) = k;

end


tmp.Mu  = MS(:,1);
tmp.S   = MS(:,2);
tmp.KL  = K;
tmp.LB  = LB;
tmp.NGmu = natgrad_lb_mu;
tmp.NGS  = natgrad_lb_S;
tmp.Sn  = Sn;
tmp.Mn  = Mn;

tmp.Mup = [mu_init;tmp.Mu];
tmp.Sp  = [S_init;tmp.S];

switch optim
    case 'mgvb'
        M = tmp;
    case 'emgvb'
        E = tmp;
    otherwise 
        P = tmp;
end

%
set_fig(1700,400,500,200)
MSp = [[mu_init,S_init];MS];

hold on
subplot(1,3,1)
plot(LB)
[~,ind] = max(LB);

hold on
subplot(1,3,2)
plot(MSp(:,2),MSp(:,1),'-')
hold on
plot(MSp(ind,2),MSp(ind,1),'xr')
plot(Sn,Mn,'o')
hold off

hold on
subplot(1,3,3)
plot(K)

%% for the contour

Nls = 15;
Nlm = 35;
lss = linspace(0.000001,0.15,Nls);
lsm = linspace(-0.1,3,Nlm);

[mgm,mgs] = meshgrid(lsm,lss);
mg = [mgm(:),mgs(:)];
vv = nan(size(mg,1),1);
for i = 1:size(mg,1)
    Vv(i,1) = get_lb(mg(i,1),mg(i,2),15000,sig2,S0,mu0,y,x);
end
V = reshape(Vv,Nls,Nlm);

%% Plot

use = P;

load('Functions\Colors.mat')
gr = col{end};
lw = 1;
lag = 10;
fs = 12;

set_fig(1700,400,500,200)

subplot(1,3,1)
plot(M.LB,'Color',col{3},'LineWidth',lw)
hold on
plot(use.LB,'Color',col{1},'LineWidth',lw)
hold off
title('Lower Bound','Interpreter','latex','FontSize',fs)
xlabel('Iteration','Interpreter','latex','FontSize',fs)
ytickformat('%.1f')
xlim([-1 MaxIter])
legend({'MGVB','MGVB rev.'},'Location','south')


subplot(1,3,2)
plot(M.KL,'Color',col{3},'LineWidth',lw)
hold on
plot(use.KL,'Color',col{1},'LineWidth',lw)
hold off
title('KL divergence','Interpreter','latex','FontSize',fs)
xlabel('Iteration','Interpreter','latex','FontSize',fs)
ytickformat('%.1f')
xlim([-1 MaxIter])


subplot(1,3,3)
contour(mgs,mgm,-log(-V),20,'ShowText','off')
hold on
plot(M.Sp,M.Mup,'-','Color',col{3},'LineWidth',lw)
plot(use.Sp,use.Mup,'-','Color',col{1},'LineWidth',lw)
plot(M.Sn,M.Mn,'or')
hold off
ylabel('$\mu$','Interpreter','latex','FontSize',fs)
xlabel('$\sigma^2$','Interpreter','latex','FontSize',fs)
title('$-\log(-\mathcal{L})$','Interpreter','latex','FontSize',fs)
ytickformat('%.1f')
% xtickformat('%.3f')
xticks([0:0.05:0.15])
xlim([0 0.15])
% axis equal

savePDF('Plots\','SLR_low')


%%

function [lb] = get_lb(mu,S,Ns,sig2,S0,mu0,y,x)
n = numel(y);
lb = nan(Ns,1);
for s = 1:Ns
    theta = normrnd(mu,sqrt(S),1,1);
    aux = (y-theta*x).^2;
    log_ll = -n/2*log(2*pi)-n/2*log(sig2)-1/2*sum(aux/sig2);
    log_q  = -1/2*log(2*pi)-1/2*log(S)-1/2*(theta-mu)^2/S;
    log_p  = -1/2*log(2*pi)-1/2*log(S0)-1/2*(theta-mu0)^2/S0;
    h = log_ll + log_p - log_q;
    lb(s,1) = h;
end

lb = mean(lb);

end
