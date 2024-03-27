wd = 'C:\Users\Martin\Desktop\EMGVB_REV';
cd(wd)
addpath(genpath(wd))

clear
clc

c1 = [0 0.4470 0.7410]; % emgvb blue
c2 = [0.8500 0.3250 0.0980]; 
c3 = [0.9290 0.6940 0.1250];%mgvb yellow
c4 = [0.4940 0.1840 0.5560];
c5 = [0.4660 0.6740 0.1880];
c6 = [0.3010 0.7450 0.9330];
c7 = [0.6350 0.0780 0.1840];
c8 = [0,102,0]/255; % mcmc green
gr = 128*ones(1,3)/255;  %gray

lw = 1.4;
fs = 12;

%% PLOT GJR Marginals - ok

close
model = 'garch_101';
load(['Vol\Save\' model '.mat'])
clc


trans = 1;

if trans
    lbl = {'$\psi_\omega$','$\psi_\alpha$','$\psi_\gamma$','$\psi_\beta$'};
else
    lbl = {'$\omega$','$\alpha$','$\gamma$','$\beta$'};
end

nPar = pMGVB.NumParams;

for i = 1:nPar
    subplot(1,nPar,i)

    col = trans+1;

    plot(Pl.emgvb{col}.x(:,i),Pl.emgvb{col}.y(:,i),'LineWidth',lw,'Color',c1)

    mu_ml = ml.tab_ml{i,col};

    hold on
    plot(Pl.mgvb{col}.x(:,i),Pl.mgvb{col}.y(:,i),'--','LineWidth',lw,'Color',c3)
    plot(Pl.mc{col}.x(:,i),Pl.mc{col}.y(:,i),':','LineWidth',1.2,'Color',c8)
    xline(mu_ml,'--r','LineWidth',1)
    title(lbl{i},'Interpreter','latex','FontSize',fs)
    hold off
end

set_fig(1700,400,700,120)

if ~trans
    leg = legend({'MGVBP','MGVB','MCMC','ML'},'Orientation','vertical','Location','southeast','NumColumns',1);
    leg.ItemTokenSize = [15,15];
    savePDF('Vol\Plots\', [model '_marginal'])
else
    savePDF('Vol\Plots\',[model '_marginal_trans'])
end



%% PLOT FIGARCH Marginals - ok

load("Vol\Save\figarch_101.mat")
close

clc

trans = 1;
if ~trans
    lbl = {'$\omega$','$\phi$','$d$','$\beta$'};
else
    lbl = {'$\psi_\omega$','$\psi_\phi$','$\psi_d$','$\psi_\beta$'};
end

nPar = 4;

for i = 1:nPar
    subplot(1,nPar,i)

    col = trans+1;

    plot(Pl.emgvb{col}.x(:,i),Pl.emgvb{col}.y(:,i),'LineWidth',lw,'Color',c1)
    
    mu_ml = ml.tab_ml{i,col};

    hold on
    plot(Pl.mgvb{col}.x(:,i),Pl.mgvb{col}.y(:,i),'--','LineWidth',lw,'Color',c3)
    plot(Pl.mc{col}.x(:,i),Pl.mc{col}.y(:,i),':','LineWidth',1.2,'Color',c8)    
    xline(mu_ml,'--r','LineWidth',1)
    title(lbl{i},'Interpreter','latex','FontSize',fs)
    hold off
    xlim([Pl.emgvb{col}.x(1,i),Pl.emgvb{col}.x(end,i)])
end

leg = legend({'MGVBP','MGVB','MCMC','ML'},'Orientation','vertical','Location','southeast','NumColumns',1);
leg.ItemTokenSize = [15,15];

set_fig(1700,400,700,120)

if ~trans
    savePDF('Vol\Plots\','figarch_101_marginal')
else
    savePDF('Vol\Plots\','figarch_101_marginal_trans')
end

%% PLOT performance / main plot for garch101 and garch 111 ok

close
model = 'garch_111';
load(['Vol\Save\' model '.mat'])
load("Colors.mat")
col1 = 3;
col2 = col1+3;

subplot(1,3,1)
plot(pEMGVB.Post.LB_smooth,'Color',c1,'LineWidth',lw)
hold on
plot(pMGVB.Post.LB_smooth,'Color',c3,'LineWidth',lw)
hold off

xlabel('Iteration','Interpreter','latex')
xlim([0 opt.maxiter])

title('Lower bound','Interpreter','latex')
xline(pEMGVB.Post.LB_indx,'-','Color',c1)
xline(pMGVB.Post.LB_indx,'-','Color',c3)
ytickformat('%.1f')

hold on
t1 = plot([nan,nan],'-','Color',col{1},'LineWidth',lw);
t2 = plot([nan,nan],'-','Color',col{3},'LineWidth',lw);
t5 = plot([nan,nan],'-','Color',col{9});
hold off

legend([t1,t2,t5],{'MGVBP','MGVB','$t^\star$','$t^\star$'},'Orientation','vertical','Location','southeast','Interpreter','latex');


subplot(1,3,2)
yyaxis left
plot(pEMGVB.Post.Perf.iter.train(:,col1),'-','Color',c2,'LineWidth',lw)
hold on
plot(pMGVB.Post.Perf.iter.train(:,col1),'--','Color',c2,'LineWidth',lw)
hold off
ytickformat('%.2f')
ax = gca;
ax.YColor  = c2;

yyaxis right
plot(pEMGVB.Post.Perf.iter.test(:,col1),'-','Color',c4,'LineWidth',lw)
xlabel('Iteration','Interpreter','latex')
ytickformat('%.3f')
hold on
plot(pMGVB.Post.Perf.iter.test(:,col1),'--','Color',c4,'LineWidth',lw)
hold off

hold on
t1 = plot([nan,nan],'-','Color',c2,'LineWidth',lw);
t2 = plot([nan,nan],'--','Color',c2,'LineWidth',lw);
t3 = plot([nan,nan],'-','Color',c4,'LineWidth',lw);
t4 = plot([nan,nan],'--','Color',c4,'LineWidth',lw);
hold off

xlim([0 opt.maxiter])
ax = gca;
ax.YColor  = c4;
title('MSE','Interpreter','latex')
legend([t1,t2,t3,t4], {'EMGVB train','MGVB train','EMGVB test','MGVB test'},'Location','northeast')

subplot(1,3,3)
yyaxis left
plot(pEMGVB.Post.Perf.iter.train(:,col2)*100,'-','Color',c2,'LineWidth',lw)
hold on
plot(pMGVB.Post.Perf.iter.train(:,col2)*100,'--','Color',c2,'LineWidth',lw)
hold off
ytickformat('%.2f')
ax = gca;
ax.YColor  = c2;

yyaxis right
plot(pEMGVB.Post.Perf.iter.test(:,col2)*100,'-','Color',c4,'LineWidth',lw)
xlabel('Iteration','Interpreter','latex')
ytickformat('%.2f')
hold on
plot(pMGVB.Post.Perf.iter.test(:,col2)*100,'--','Color',c4,'LineWidth',lw)
hold off

hold on
t1 = plot([nan,nan],'-','Color',c2,'LineWidth',lw);
t2 = plot([nan,nan],'--','Color',c2,'LineWidth',lw);
t3 = plot([nan,nan],'-','Color',c4,'LineWidth',lw);
t4 = plot([nan,nan],'--','Color',c4,'LineWidth',lw);
hold off

xlim([0 opt.maxiter])
ax = gca;
ax.YColor  = c4;
legend([t1,t2,t3,t4], {'EMGVB train','MGVB train','EMGVB test','MGVB test'})
title('Qlik ($\times 10^2$)','Interpreter','latex')


set_fig(1700,400,700,150)
savePDF('Vol\Plots\',[model '_perf'])

%%
