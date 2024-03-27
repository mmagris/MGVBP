function[out] = garch_perf(obj,data,mu)

[nll,ht] = garch_nll_ht(data(:,1),mu,obj);

mse     = @(col) mean((data(:,col).^2-ht).^2);
qlik    = @(col) mean(data(:,col).^2./ht-log(data(:,col).^2./ht)-1);
out     = [-nll,mse(1),mse(2),mse(3),qlik(1),qlik(2),qlik(3)];

end