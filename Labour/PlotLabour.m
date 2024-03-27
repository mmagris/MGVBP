wd = 'C:\Users\Martin\Desktop\EMGVB_REV';
cd(wd)
addpath(genpath(wd))

clear
clc

load("Labour\Results\Results.mat")
load("Colors.mat")
gr = col{end};
lw = 1.4;
lag = 10;
fs = 12;

%%

set_fig(1700,100,500,400)
set_fig(1700,400,700,200)

tiledlayout(1,4)

nexttile
f = @(x,i) (movmean(x.Post.train.perf_theta(:,i),lag));

semilogy(f(pEMGVB,1),'Color',col{1},'LineWidth',lw);
hold on
semilogy(f(pEMGVB,2),'Color',col{2},'LineWidth',lw);
semilogy(f(pEMGVB,3),'Color',col{3},'LineWidth',lw);
semilogy(f(pEMGVB,4),'Color',col{4},'LineWidth',lw);
hold off

ylim([0.62 0.73])
xlim([0 1000])
xlabel('Iteration','Interpreter','latex')
ylabel('Performance','Interpreter','latex')
ytickformat('%.2f')
title('MGVBP train','FontWeight','normal')
set(gca, 'YGrid', 'on', 'XGrid', 'off')



nexttile
f = @(x,i) (movmean(x.Post.train.perf_theta(:,i),lag));

semilogy(f(pMGVB,1),'Color',col{1},'LineWidth',lw);
hold on
semilogy(f(pMGVB,2),'Color',col{2},'LineWidth',lw);
semilogy(f(pMGVB,3),'Color',col{3},'LineWidth',lw);
semilogy(f(pMGVB,4),'Color',col{4},'LineWidth',lw);
hold off

ylim([0.62 0.73])
xlim([0 1000])
xlabel('Iteration','Interpreter','latex')
ylabel('Performance','Interpreter','latex')
ytickformat('%.2f')
title('MGVB train','FontWeight','normal')
set(gca, 'YGrid', 'on', 'XGrid', 'off')






nexttile
f = @(x,i) (movmean(x.Post.test.perf_theta(:,i),lag));

semilogy(f(pEMGVB,1),'Color',col{1},'LineWidth',lw);
hold on
semilogy(f(pEMGVB,2),'Color',col{2},'LineWidth',lw);
semilogy(f(pEMGVB,3),'Color',col{3},'LineWidth',lw);
semilogy(f(pEMGVB,4),'Color',col{4},'LineWidth',lw);
hold off
grid
ylim([0.55 0.73])
xlim([0 1000])
xlabel('Iteration','Interpreter','latex')
ylabel('Performance','Interpreter','latex')
ytickformat('%.2f')
title('MGVBP test','FontWeight','normal')
set(gca, 'YGrid', 'on', 'XGrid', 'off')


nexttile
f = @(x,i) (movmean(x.Post.test.perf_theta(:,i),lag));

semilogy(f(pMGVB,1),'Color',col{1},'LineWidth',lw);
hold on
semilogy(f(pMGVB,2),'Color',col{2},'LineWidth',lw);
semilogy(f(pMGVB,3),'Color',col{3},'LineWidth',lw);
semilogy(f(pMGVB,4),'Color',col{4},'LineWidth',lw);
hold off


ylim([0.55 0.73])
xlim([0 1000])
xlabel('Iteration','Interpreter','latex')
ylabel('Performance','Interpreter','latex')
ytickformat('%.2f')
title('MGVB test','FontWeight','normal')
set(gca, 'YGrid', 'on', 'XGrid', 'off')

hold on
t3 = plot([nan,nan],'-','Color',col{1},'LineWidth',lw) ;
t4 = plot([nan,nan],'-','Color',col{2},'LineWidth',lw) ;
t5 = plot([nan,nan],'-','Color',col{3},'LineWidth',lw) ;
t6 = plot([nan,nan],'-','Color',col{4},'LineWidth',lw) ;
hold off

leg = legend([t3,t4,t5,t6],{'Accuracy','Precision','Recall','f1'},'Orientation','horizontal');
leg.Layout.Tile = 'south';

savePDF('Labour\Plots\','Labour_Perf')


%%

close
set_fig(1700,400,700,200)

subplot(1,2,1)

plot(pEMGVB.Post.iter.mu(:,1:1),'Color',col{1},'LineWidth',lw)
hold on
plot(pEMGVB_h.Post.iter.mu(:,1:1),'--','Color',col{1},'LineWidth',lw)
plot(pEMGVB_h_d.Post.iter.mu(:,1:1),':','Color',col{1},'LineWidth',lw)

plot(pEMGVB.Post.iter.mu(:,2),'Color',col{2},'LineWidth',lw)
plot(pEMGVB_h.Post.iter.mu(:,2),'--','Color',col{2},'LineWidth',lw)
plot(pEMGVB_h_d.Post.iter.mu(:,2),':','Color',col{2},'LineWidth',lw)

plot(pEMGVB.Post.iter.mu(:,3),'Color',col{3},'LineWidth',lw)
plot(pEMGVB_h.Post.iter.mu(:,3),'--','Color',col{3},'LineWidth',lw)
plot(pEMGVB_h_d.Post.iter.mu(:,3),':','Color',col{3},'LineWidth',lw)

plot(pEMGVB.Post.iter.mu(:,4),'Color',col{8},'LineWidth',lw)
plot(pEMGVB_h.Post.iter.mu(:,4),'--','Color',col{8},'LineWidth',lw)
plot(pEMGVB_h_d.Post.iter.mu(:,4),':','Color',col{8},'LineWidth',lw)

hold off

hold on
t1 = plot([nan,nan],'-','Color',gr,'LineWidth',lw) ;
t2 = plot([nan,nan],'--','Color',gr,'LineWidth',lw) ;
t3 = plot([nan,nan],':','Color',gr,'LineWidth',lw) ;
t4 = plot([nan,nan],'-','Color',col{1},'LineWidth',lw) ;
t5 = plot([nan,nan],'-','Color',col{2},'LineWidth',lw) ;
t6 = plot([nan,nan],'-','Color',col{3},'LineWidth',lw) ;
t7 = plot([nan,nan],'-','Color',col{8},'LineWidth',lw) ;
hold off


legend([t4,t5,t6,t7],...
    {'$\mu_{1}$','$\mu_{2}$','$\mu_{3}$','$\mu_{4}$'},...
    'Interpreter','latex','NumColumns',2,'Location','southeast')

grid
xlabel('Iteration','Interpreter','latex')
ylabel('Posterior means','Interpreter','latex')
xlim([0 1000])


subplot(1,2,2)

plot(pEMGVB.Post.iter.SigInv(:,1:1),'Color',col{1},'LineWidth',lw)
hold on
plot(pEMGVB_h.Post.iter.SigInv(:,1:1),'--','Color',col{1},'LineWidth',lw)
plot(pEMGVB_h_d.Post.iter.SigInv(:,1:1),':','Color',col{1},'LineWidth',lw)

plot(pEMGVB.Post.iter.SigInv(:,10),'Color',col{2},'LineWidth',lw)
plot(pEMGVB_h.Post.iter.SigInv(:,10),'--','Color',col{2},'LineWidth',lw)
plot(pEMGVB_h_d.Post.iter.SigInv(:,2),':','Color',col{2},'LineWidth',lw)


plot(pEMGVB.Post.iter.SigInv(:,19),'Color',col{3},'LineWidth',lw)
plot(pEMGVB_h.Post.iter.SigInv(:,19),'--','Color',col{3},'LineWidth',lw)
plot(pEMGVB_h_d.Post.iter.SigInv(:,3),':','Color',col{3},'LineWidth',lw)


plot(pEMGVB.Post.iter.SigInv(:,2),'Color',col{4},'LineWidth',lw)
plot(pEMGVB_h.Post.iter.SigInv(:,2),'--','Color',col{4},'LineWidth',lw)

plot(pEMGVB.Post.iter.SigInv(:,3),'Color',col{5},'LineWidth',lw)
plot(pEMGVB_h.Post.iter.SigInv(:,3),'--','Color',col{5},'LineWidth',lw)

plot(pEMGVB.Post.iter.SigInv(:,18),'Color',col{6},'LineWidth',lw)
plot(pEMGVB_h.Post.iter.SigInv(:,18),'--','Color',col{6},'LineWidth',lw)
hold off

hold on
t1 = plot([nan,nan],'-','Color',gr,'LineWidth',lw) ;
t2 = plot([nan,nan],'--','Color',gr,'LineWidth',lw) ;
t3 = plot([nan,nan],':','Color',gr,'LineWidth',lw) ;
t4 = plot([nan,nan],'-','Color',col{1},'LineWidth',lw) ;
t5 = plot([nan,nan],'-','Color',col{2},'LineWidth',lw) ;
t6 = plot([nan,nan],'-','Color',col{3},'LineWidth',lw) ;
t7 = plot([nan,nan],'-','Color',col{4},'LineWidth',lw) ;
t8 = plot([nan,nan],'-','Color',col{5},'LineWidth',lw); 
t9 = plot([nan,nan],'-','Color',col{6},'LineWidth',lw); 
hold off
set(gca, 'YGrid', 'on', 'XGrid', 'off')


legend([t4,t5,t6,t7,t8,t9],...
    '$\Sigma_{11}$','$\Sigma_{22}$','$\Sigma_{33}$','$\Sigma_{12}$','$\Sigma_{13}$','$\Sigma_{23}$',...
    'Interpreter','latex','NumColumns',2)

ylim([-0 650])
xlabel('Iteration','Interpreter','latex')
ylabel('Posterior covariances','Interpreter','latex')
xlim([0 1000])


savePDF('Labour\Plots\','Labour_PostMeanCov')

%%
close all
set_fig(1700,400,700,150)
xp = 31:opt.MaxIter;

subplot(1,3,1)
plot(xp,pEMGVB.Post.LB_smooth,'Color',col{1},'LineWidth',lw)
hold on
plot(xp,pEMGVB_h.Post.LB_smooth,'--','Color',col{2},'LineWidth',lw) 
plot(xp,pMGVB.Post.LB_smooth,'Color',col{3},'LineWidth',lw)
plot(xp,pMGVB_h.Post.LB_smooth,'--','Color',col{4},'LineWidth',lw)

N = 970;
[~,indx] = max(pEMGVB.Post.LB_smooth(1:N));
xline(indx,'-','Color',col{1})

[~,indx] = max(pEMGVB_h.Post.LB_smooth(1:N));
xline(indx,'--','Color',col{2})

[~,indx] = max(pMGVB_h.Post.LB_smooth(1:N));
xline(indx,'-','Color',col{3})

[~,indx] = max(pMGVB_h.Post.LB_smooth(1:N));
xline(indx,'--','Color',col{4})

hold off
% grid
xlabel('Iteration','Interpreter','latex')
title('Lower bound','Interpreter','latex')

ylim([-400,-355])
xlim([0 1000])

hold on
t1 = plot([nan,nan],'-','Color',col{1},'LineWidth',lw) ;
t2 = plot([nan,nan],'--','Color',col{2},'LineWidth',lw) ;
t3 = plot([nan,nan],'-','Color',col{3},'LineWidth',lw) ;
t4 = plot([nan,nan],'--','Color',col{4},'LineWidth',lw) ;
t5 = plot([nan,nan],'-','Color',gr,'LineWidth',0.8) ;
hold off

leg = legend([t1,t2,t3,t4,t5],{'MGVBP','MGVBP$^{h-func.}$','MGVB','MGVB$^{h-func.}$','$t^\star$'},...
    'Interpreter','latex','Location','southeast');
% leg.ItemTokenSize = [15,15];


sml = diag(lm.glm.covb);

% nexttile

for i = 1:2

    mu1_i   = pMGVB.Post.mu(i);
    s1_i    = pMGVB.Post.Sig2(i)^0.5;

    mu2_i   = pEMGVB.Post.mu(i);
    s2_i    = pEMGVB.Post.Sig2(i)^0.5;

    muml_i  = lm.glm.beta(i);
    sml_i   = sml(i);

    xp = linspace(mu1_i-4*s1_i,mu1_i+4*s1_i,100);
    yp1 = normpdf(xp,mu1_i,s1_i);
    yp2 = normpdf(xp,mu2_i,s2_i);
    ypml = normpdf(xp,muml_i,sml_i);

    xx = mcmc.mean(i)-4*mcmc.std(i):0.002:mcmc.mean(i)+4*mcmc.std(i);
    yy_mcmc = ksdensity(mcmc.chain(:,i),xx,'Bandwidth',0.022);

    subplot(1,3,1+i)
    
    plot(xp,yp2,'-','LineWidth',lw,'Color',col{1})
    hold on
    plot(xp,yp1,'--','LineWidth',lw,'Color',col{3})
    plot(xx,yy_mcmc,':','LineWidth',lw,'Color',col{8})
    xline(muml_i,'--r','LineWidth',1)
    hold off
    title(['$\beta_' num2str(i-1) '$'],'Interpreter','latex','FontSize',fs)
    leg = legend({'MGVBP','MGVB','MCMC','ML'},'Orientation','vertical','Location','northeast','NumColumns',1);
    leg.ItemTokenSize = [20,5];
    ylim([0 2.1])
    if i == 1        
        xlim([-0.1 1.5])
    end

end

savePDF('Labour\Plots\','Labour_LBmargins')

