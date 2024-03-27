function [h_func,llh,log_prior] = h_func_labour(data,theta,setting)

% Extract additional settings
d = length(theta);
Sig = setting.Prior.Sig;
mu = setting.Prior.Mu;

% Extract data
X = data(:,1:end-1);
y = data(:,end);

% Compute log likelihood
aux = X*theta;
llh = y.*aux-log(1+exp(aux));
llh = sum(llh);


% Compute log prior
aux = (theta-mu);

if numel(Sig) == 1
    log_prior = -d/2*log(2*pi)-d/2*sum(log(Sig))-1/2*aux'*(aux./Sig); % scalar precision
elseif numel(Sig) == d
    log_prior = -d/2*log(2*pi)-1/2*sum(log(Sig))-1/2*aux'*(aux./Sig); % vector
else
    log_prior = -d/2*log(2*pi)-1/2*log(det(Sig))-1/2*aux'*(Sig\aux);  % matrix
end

% Compute h(theta) = log p(y|theta) + log p(theta)
h_func = llh + log_prior;


end