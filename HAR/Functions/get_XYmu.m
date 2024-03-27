function[X,Y,m,u,ll,mse,qlik] = get_XYmu(data,mu)
m = mu(1:end-1);
u = exp(mu(end));
X = data(:,2:end);
Y = data(:,1);

d_theta = size(mu,1);
N = size(Y,1);

Yh = X*m;
ll   = -0.5*sum(log(2*pi*u^2)+(Y-Yh).^2./u^2);
mse  = sum((Y-Yh).^2)*1/(N-(d_theta-1));
qlik = mean(Y./Yh - log(Y./Yh) -1 );
end