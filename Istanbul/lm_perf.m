
function[out] = lm_perf(Data,theta,sig,isStruct)

if nargin == 3
    isStruct = true;
end

Y = Data.train(:,1);
X = Data.train(:,2:end);
perf.ll_train = -0.5*sum(log(2*pi*sig^2)+(Y-X*theta).^2./sig^2);
perf.mse_train = sum((Y-X*theta).^2)*1/(Data.n.train-4);

Y = Data.test(:,1);
X = Data.test(:,2:end);
perf.mse_test = sum((Y-X*theta).^2)*1/(Data.n.train-4);
perf.ll_test = -0.5*sum(log(2*pi*sig^2)+(Y-X*theta).^2./sig^2);

vec = [perf.ll_train,perf.mse_train,perf.ll_test,perf.mse_test];

if isStruct
    out = perf;
else
    out = vec;
end

end