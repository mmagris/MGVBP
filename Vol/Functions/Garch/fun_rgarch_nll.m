function[nll,ht] = fun_rgarch_nll(data,par,p,q,isTrans)

nll = rgarch_likelihood(par,p,q,data(:,1),data(:,2),0,isTrans);
ht  = nan;

end