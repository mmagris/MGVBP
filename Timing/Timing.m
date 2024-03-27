
wd = 'C:\Users\Martin\Desktop\EMGVB_REV\Timing';
cd(wd)
addpath(genpath('SLR'))


clear
clc
close all

n = 100;
k = 150;
X = [ones(n,1),rand(n,k-1)];
b = rand(k,1);
sig2 = 1;
y = X*b + normrnd(0,sqrt(sig2),n,1);

% fitlm(X,y,'linear','Intercept',false)

mu0 = ones(k,1);
S0  = 1*eye(k);
iS0 = inv(S0);
% ret = @(S,xi) S + xi +0.5*xi*inv(S)*xi;

%%

clc


MaxIter = 100;
Ns = 100;
useLL = 1;

Nt = 25;
T = zeros(Nt,2);

K = [2,3,5,10,15,25,50,100,150];

for j = 1:numel(K)
    k = K(j);
    mu0 = ones(k,1);
    S0  = 1*eye(k);
    iS0 = inv(S0);
    for i = 1:Nt
        [tmgvb,LB1]  = wrapper(MaxIter,Ns,'mgvb',k,y,X(:,1:k),useLL,sig2,S0,iS0,mu0);
        [tmgvbp,LB2] = wrapper(MaxIter,Ns,'mgvbp',k,y,X(:,1:k),useLL,sig2,S0,iS0,mu0);
        T(i,:) = [tmgvb,tmgvbp];
        fprintf('K: %i, Iteration: %i.\n',k,i)
    end

    plot([LB1,LB2])
    m = mean(T,1);

    M(j,:) = [m,1-m(2)/m(1)];

end


save('Timing.mat','M','K','Ns')

%%



function[tout,LB] = wrapper(MaxIter,Ns,optim,k,y,X,useLL,sig2,S0,iS0,mu0)


mu_init = zeros(k,1);
S_init  = 0.1*eye(k);
beta = 0.000002;
mu = mu_init;
S  = S_init;
iS = inv(S);
ret = @(S,iS,xi) S + xi +0.5*xi*iS*xi;
LB = nan(MaxIter,1);

tic
for iter = 1:MaxIter

    [lb,grad_lb_mu,grad_lb_S] = get_lb_grad(mu,S,iS,Ns,sig2,S0,iS0,mu0,y,X,useLL);

    beta_use = beta;

    natgrad_lb_mu = S*grad_lb_mu;

    mu = mu +beta_use*natgrad_lb_mu;

    switch optim
        case 'mgvb'
            natgrad_lb_S  = 2*S*grad_lb_S*S;
            S  = ret(S,iS,beta_use*natgrad_lb_S);
            iS = inv(S);
        case 'mgvbp'
            natgrad_lb_S  = -grad_lb_S;
            iS = ret(iS,S,beta_use*natgrad_lb_S);
            S = inv(iS);
    end

    LB(iter,1) = lb;

end

tout = toc;

end

function[lb,grad_lb_mu,grad_lb_S] = get_lb_grad(mu,S,iS,Ns,sig2,S0,iS0,mu0,y,X,useLL)

k = numel(mu);
n = numel(y);

grad_lb_mu = nan(k,Ns);
grad_lb_S = nan(k^2,Ns);
lb = nan(1,Ns);

theta = mvnrnd(mu,S,Ns)';
log_q = logmvnpdf(theta',mu',S);
log_p = logmvnpdf(theta',mu0',S0);

for s = 1:Ns
    th = theta(:,s);

    aux = (y-X*th).^2;
    log_ll = -n/2*log(2*pi)-n/2*log(sig2)-1/2*sum(aux/sig2);


    aux = iS*(th-mu);
    grad_log_q_mu = aux;
    grad_log_q_S  = -1/2*(iS - aux*(aux'));
    h = log_ll + log_p(s) - log_q(s);

    lb(1,s) = h;

    if ~useLL
        f = h;
    else
        f = log_ll;
    end

    grad_lb_mu(:,s)  = grad_log_q_mu*f;
    grad_lb_S(:,s)   = grad_log_q_S(:)*f;

end

lb = mean(lb,2);
gS = reshape(mean(grad_lb_S,2),k,k);

if ~useLL
    grad_lb_mu  = mean(grad_lb_mu,2);
    grad_lb_S   = gS;
else
    grad_lb_mu  = mean(grad_lb_mu,2) + S*iS0*(mu-mu0);
    grad_lb_S   = gS + 1/2*(iS-iS0);
end

end



function [logp] = logmvnpdf(x,mu,Sigma)
% outputs log likelihood array for observations x  where x_n ~ N(mu,Sigma)
% x is NxD, mu is 1xD, Sigma is DxD
[N,D] = size(x);
const = -0.5 * D * log(2*pi);
xc = bsxfun(@minus,x,mu);
term1 = -0.5 * sum((xc / Sigma) .* xc, 2); % N x 1
term2 = const - 0.5 * logdet(Sigma);    % scalar
logp = term1' + term2;
end
function y = logdet(A)
U = chol(A);
y = 2*sum(log(diag(U)));
end
