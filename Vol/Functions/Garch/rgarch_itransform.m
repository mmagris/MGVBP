function[out] = rgarch_itransform(upar,p,q)

% Maps real line (unconstrained) to constrained parameters that make sense

f = @(x) exp(x)./(1+exp(x));
k = p+q;

umu      = upar(1);
uomega   = upar(2) ;

ua          = upar(3:3+q-1)';
ub          = upar(3+q:k+2)';
ueta1       = upar(3+k);
ueta2       = upar(4+k);
ud          = upar(5+k);
ulambda     = upar(6+k);
uxi         = upar(7+k);

alpha       = f(ua).*(1-f(ub));
beta        = f(ua).*f(ub);
delta       = 1/2*f(ud)+3/4;
xi          = f(uxi)-1/2;

mu      = umu;
omega   = uomega;
eta1    = ueta1;
eta2    = ueta2;
lambda  = ulambda;

out = [mu,omega,alpha,beta,eta1,eta2,delta,lambda,xi]';

end