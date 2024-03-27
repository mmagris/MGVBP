function[out] = rgarch_transform(par,p,q)

% constrained parameters to the real line (unconstrained)

k = p+q;

mu     = par(1);
omega  = par(2);
a      = par(3:3+q-1)';
b      = par(3+q:k+2)';
eta1   = par(3+k);
eta2   = par(4+k);
d      = par(5+k);
lam    = par(6+k);
xi     = par(7+k);

umu      = mu;
uomega   = omega;
ueta1    = eta1;
ueta2    = eta2;
ulambda  = lam;

ua      = log(-(a+b)/(a+b-1));
ub      = log(b/a);
ud      = log(-(4*d-3)/(4*d-5));
uxi     = log(-(xi+1/2)/(xi-1/2));

out = [umu,uomega,ua,ub,ueta1,ueta2,ud,ulambda,uxi]';

end