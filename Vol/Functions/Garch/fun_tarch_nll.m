function[nll,ht] = fun_tarch_nll(data,par,p,o,q,isTrans)

epsilon = data;
m  =  max([p o q]);

fepsilon            =  [mean(epsilon.^2)*ones(m,1) ; epsilon.^2];
fIepsilon           =  [0.5*mean(epsilon.^2)*ones(m,1) ; epsilon.^2.*(epsilon<0)];

epsilon_augmented   = [zeros(m,1);epsilon];
T                   = size(fepsilon,1);

error_type = 1;
tarch_type = 2;
back_cast = 0.7;

[nll,~,ht] = tarch_likelihood(par,epsilon_augmented,fepsilon,fIepsilon,p,o,q,error_type,tarch_type,back_cast,T,isTrans);

end

