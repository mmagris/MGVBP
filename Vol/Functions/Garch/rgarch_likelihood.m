function[nLL,Lr,Lrh] = rgarch_likelihood(par,p,q,y,r,byObs,isTrans)

if nargin == 5
    byObs = 0;
end

if isTrans
    par = rgarch_itransform(par,p,q);
end


k       = p+q;
mu      = par(1);
omega   = par(2);
alpha   = par(3:3+q-1)';
beta    = par(3+q:k+2)';
eta1    = par(3+k);
eta2    = par(4+k);
delta   = par(5+k);
lambda  = par(6+k);
xi      = par(7+k);


N       = numel(y);
log_s   = nan(N,1);
z       = log_s;
u       = log_s;
tau     = @(z) eta1*z+eta2*(z^2-1);


log_s(1) = log(var(y));
z(1)     = y(1)/exp(log_s(1))^0.5;
u(1)     = log(r(1)) - (xi + delta*log_s(1)+tau(z(1)));


for j = 2:N
    a = 0; b = 0;
    
    for i = 1:min(q,max(j-1,1))
        a = a + alpha(i)*log(r(j-i));
    end
    
    for i = 1:min(p,max(j-1,1))
        b = b + beta(i)*log_s(j-i);
    end
    
    log_s(j)  = omega + a + b;
    z(j)      = (y(j)-mu)/exp(log_s(j))^0.5;
    u(j)      = log(r(j)) - (xi + delta*log_s(j)+tau(z(j)));
end


if byObs == 0
    Lr  = -1/2*sum( log(2*pi) + log_s + (y-mu).^2./exp(log_s) );
    Lrh = -1/2*sum( log(2*pi) + log(lambda^2)+u.^2/lambda^2 );
    LL  = Lr+Lrh;
else
    Lr  = -1/2*(log(2*pi) + log_s + (y-mu).^2./exp(log_s)) ;
    Lrh = -1/2*(log(2*pi) + log(lambda^2)+u.^2/lambda^2) ;
    LL  = Lr+Lrh;
end
nLL = -LL;

end

