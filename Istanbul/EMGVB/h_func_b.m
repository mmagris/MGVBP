function [h_func,llh,log_prior] = h_func_b(data,theta,setting)

% Extract additional settings
Sig = setting.Prior.Sig;
mu  = setting.Prior.Mu;
B = setting.Block;

% Extract data
y = data(:,1);
x = data(:,2:end);

% Compute log likelihood
b = theta(1:end-1);
s = exp(theta(end));

llh = -0.5*sum(log(2*pi*s^2)+(y-x*b).^2./s^2);


log_prior = 0;
aux = (theta-mu);


if ~isnan(setting.iFullPrior)
    % Relevant when estimating a block-diagonal posterior,
    % with the prior being full or with blocks different than those of the
    % posterior
    d           = length(theta);
    iS0         = setting.iFullPrior;
    log_prior   = -d/2*log(2*pi)+1/2*log(det(iS0))-1/2*aux'*(iS0*aux);
else

    for b = 1:B.n
        indx = B.indx{b,1};
        type = setting.Sig0_type(b);

        iS0 = setting.Sig_inv_0{b};

        if type == 3
            f_log_prior = @(aux,Sig,d) -d/2*log(2*pi)-d/2*sum(log(Sig))-1/2*aux'*(aux./Sig); % scalar precision
        elseif type == 2
            f_log_prior = @(aux,Sig,d) -d/2*log(2*pi)-1/2*sum(log(Sig))-1/2*aux'*(aux./Sig); % vector
        elseif type == 1
          % f_log_prior = @(aux,Sig,d) -d/2*log(2*pi)-1/2*log(det(Sig))-1/2*aux'*(Sig\aux);  % matrix
            f_log_prior = @(aux,Sig,d) -d/2*log(2*pi)+1/2*log(det(iS0))-1/2*aux'*(iS0*aux);  % matrix
        end

        log_prior = log_prior + f_log_prior(aux(indx),Sig{b},B.blks(b));

    end

end

% Compute h(theta) = log p(y|theta) + log p(theta)
h_func = llh + log_prior;


end
