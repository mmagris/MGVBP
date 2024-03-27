classdef MGVBm < VBayesLab
    %MVB Summary of this class goes here
    %   Detailed explanation goes here

    properties
        GradClipInit       % If doing gradient clipping at the beginning
    end

    methods
        function obj = MGVBm(mdl,data,varargin)
            %MVB Construct an instance of this class
            %   Detailed explanation goes here
            obj.Method        = 'MGVBm';
            obj.GradWeight    = 0.4;    % Small gradient weight is better
            obj.GradClipInit  = 0;      % Sometimes we need to clip the gradient early

            % Parse additional options
            if nargin > 2
                paramNames = {'NumSample'             'LearningRate'       'GradWeight'      'GradClipInit' ...
                    'MaxIter'               'MaxPatience'        'WindowSize'      'Verbose' ...
                    'InitMethod'            'StdForInit'         'Seed'            'MeanInit' ...
                    'SigInitScale'          'LBPlot'             'GradientMax' ...
                    'NumParams'             'DataTrain'          'Setting'         'StepAdaptive' ...
                    'SaveParams'};
                paramDflts = {obj.NumSample            obj.LearningRate    obj.GradWeight    obj.GradClipInit ...
                    obj.MaxIter              obj.MaxPatience     obj.WindowSize    obj.Verbose ...
                    obj.InitMethod           obj.StdForInit      obj.Seed          obj.MeanInit ...
                    obj.SigInitScale         obj.LBPlot          obj.GradientMax  ...
                    obj.NumParams            obj.DataTrain       obj.Setting       obj.StepAdaptive ...
                    obj.SaveParams};

                [obj.NumSample,...
                    obj.LearningRate,...
                    obj.GradWeight,...
                    obj.GradClipInit,...
                    obj.MaxIter,...
                    obj.MaxPatience,...
                    obj.WindowSize,...
                    obj.Verbose,...
                    obj.InitMethod,...
                    obj.StdForInit,...
                    obj.Seed,...
                    obj.MeanInit,...
                    obj.SigInitScale,...
                    obj.LBPlot,...
                    obj.GradientMax,...
                    obj.NumParams,...
                    obj.DataTrain,...
                    obj.Setting,...
                    obj.StepAdaptive,...
                    obj.SaveParams] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});
            end

            % Check if model object or function handle is provided
            if (isobject(mdl)) % If model object is provided
                obj.Model = mdl;
                obj.ModelToFit = obj.Model.ModelName; % Set model name if model is specified
            else % If function handle is provided
                obj.HFuntion = mdl;
            end

            % Main function to run MGVB
            obj.Post   = obj.fit(data);
        end

        %% VB main function
        function Post = fit(obj,data)

            % Extract model object if provided
            if (~isempty(obj.Model))
                model           = obj.Model;
                d_theta         = model.NumParams;      % Number of parameters
            else  % If model object is not provided, number of parameters must be provided
                if (~isempty(obj.NumParams))
                    d_theta = obj.NumParams;
                else
                    error('Number of model parameters have to be specified!')
                end
            end

            % Extract sampling setting
            std_init        = obj.StdForInit;
            eps0            = obj.LearningRate;
            S               = obj.NumSample;
            ini_mu          = obj.MeanInit;
            window_size     = obj.WindowSize;
            max_patience    = obj.MaxPatience;
            momentum_weight = obj.GradWeight;
            init_scale      = obj.SigInitScale;
            stepsize_adapt  = obj.StepAdaptive;
            max_iter        = obj.MaxIter;
            lb_plot         = obj.LBPlot;
            max_grad        = obj.GradientMax;
            max_grad_init   = obj.GradClipInit;
            hfunc           = obj.HFuntion;
            setting         = obj.Setting;
            verbose         = obj.Verbose;
            save_params     = obj.SaveParams;

            setting.d_theta = d_theta;
            [setting, isDiag, useHfunc, ~] = check_setting(setting);
            [setting, ~] = Factor_settings(setting);

            % Store some variables at each iteration
            ssave = create_iter_struct(save_params,max_iter,setting);

            % Initialization
            iter        = 0;
            patience    = 0;
            stop        = false;
            LB_smooth   = zeros(1,max_iter+1);
            LB          = zeros(1,max_iter+1);

            if isempty(ini_mu)
                mu = normrnd(0,std_init,d_theta,1);
            else
                mu = ini_mu;
            end

            % Initialize Sig_inv
            if ~isDiag
                I       = eye(d_theta);
                n_par   = d_theta+d_theta*d_theta;
                Sig     = init_scale*I;
            else
                u       = ones(d_theta,1);
                n_par   = d_theta+d_theta;
                Sig     = u.*init_scale;
            end

            % Functions, for Sig full or diagonal
            if ~isDiag
                sim_theta    = @(s,mu,C_lower,rqmc) mu + C_lower*rqmc(s,:)';
                log_pdf      = @(theta,mu,Sig_inv) -d_theta/2*log(2*pi)+1/2*log(det(Sig_inv))-1/2*(theta-mu)'*Sig_inv*(theta-mu);
                fun_gra_log_q_Sig = @(aux,Sig_inv) -1/2*Sig_inv+1/2*Sig_inv*aux*(aux')*Sig_inv;
            else
                sim_theta    = @(s,mu,C_lower,rqmc) mu + C_lower.*rqmc(s,:)';
                log_pdf      = @(theta,mu,Sig_inv) -d_theta/2*log(2*pi)+1/2*sum(log(Sig_inv))-1/2*(theta-mu)'*((theta-mu).*Sig_inv);
                fun_gra_log_q_Sig = @(aux,Sig_inv) -1/2.*Sig_inv+1/2*Sig_inv.*aux.*(aux).*Sig_inv;
            end

            mk_L_Sig = @(Sig_inv) get_L_Sig(Sig_inv,setting);

            if setting.doCV == 1
                fun_cv = @(A,B) control_variates(A,B,S,1);
            else
                fun_cv = @(A,B) control_variates(A,B,S,0);
            end

            c12                         = zeros(1,n_par);
            gra_log_q_lambda            = zeros(S,n_par);
            grad_log_q_h_function       = zeros(S,n_par);
            grad_log_q_h_function_cv    = zeros(S,n_par);
            lb_log_h                    = zeros(S,1);

            if save_params
                llh_s = zeros(S,1);
                log_q_lambda_s = zeros(S,1);
            end

            [L,Sig_inv] = mk_L_Sig(Sig);

            rqmc = utils_normrnd_qmc(S,d_theta);

            for s = 1:S
                % Parameters in Normal distribution
                theta = sim_theta(s,mu,L,rqmc);

                [h_theta,llh] = hfunc(data,theta,setting);

                % Log q_lambda
                log_q_lambda = log_pdf(theta,mu,Sig_inv);

                % h function
                h_function = h_theta - log_q_lambda;

                if useHfunc
                    f = h_function;
                else
                    f = llh;
                end

                if save_params
                    llh_s(s,1) = llh;
                    log_q_lambda_s(s,1) = log_q_lambda;
                end

                % To compute the lowerbound
                lb_log_h(s) = h_function;

                aux                           = theta-mu;
                gra_log_q_mu                  = aux;
                gra_log_q_Sig                 = fun_gra_log_q_Sig(aux,Sig_inv);
                gra_log_q_lambda(s,:)         = [gra_log_q_mu;gra_log_q_Sig(:)]';
                grad_log_q_h_function(s,:)    = gra_log_q_lambda(s,:)*f;
                grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(f-c12);
            end

            c12 = fun_cv(grad_log_q_h_function,gra_log_q_lambda);
            Y12 = mean(grad_log_q_h_function_cv)';
            Y12 = grad_clipping(Y12,max_grad_init);

            [gradLB_mu,gradLB_Sig] = nat_gradients(setting,Y12,mu,Sig,Sig_inv);

            gradLB_Sig_momentum    = gradLB_Sig;
            gradLB_mu_momentum     = gradLB_mu;

            LB0 = mean(lb_log_h);
            if verbose ~=0
                disp(['Iter: 0000 |LB: ', num2str(LB0)])
            end

            % Prepare for the next iterations
            mu_best = mu;
            Sig_best = Sig;

            while ~stop

                iter = iter+1;
                if iter>stepsize_adapt
                    stepsize = eps0*stepsize_adapt/iter;
                else
                    stepsize = eps0;
                end

                Sig_inv_old = Sig_inv;

                mu      = mu + stepsize*gradLB_mu_momentum;
                Sig     = obj.retraction_spd(Sig, Sig_inv, gradLB_Sig_momentum, stepsize);

                [L,Sig_inv] = mk_L_Sig(Sig);

                if save_params
                    llh_s = zeros(S,1);
                    log_q_lambda_s = zeros(S,1);
                end

                rqmc     = utils_normrnd_qmc(S,d_theta);


                for s = 1:S
                    % Parameters in Normal distribution
                    theta = sim_theta(s,mu,L,rqmc);

                    [h_theta,llh] = hfunc(data,theta,setting);

                    % log q_lambda
                    log_q_lambda = log_pdf(theta,mu,Sig_inv);

                    % h function
                    h_function = h_theta - log_q_lambda;

                    if useHfunc
                        f = h_function;
                    else
                        f = llh;
                    end

                    if save_params
                        llh_s(s,1) = llh;
                        log_q_lambda_s(s,1) = log_q_lambda;
                    end

                    % To compute the lowerbound
                    lb_log_h(s) = h_function;

                    aux                           = theta-mu;
                    gra_log_q_mu                  = aux;
                    gra_log_q_Sig                 = fun_gra_log_q_Sig(aux,Sig_inv);
                    gra_log_q_lambda(s,:)         = [gra_log_q_mu;gra_log_q_Sig(:)]';
                    grad_log_q_h_function(s,:)    = gra_log_q_lambda(s,:)*f;
                    grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(f-c12);
                end


                c12 = fun_cv(grad_log_q_h_function,gra_log_q_lambda);
                Y12 = mean(grad_log_q_h_function_cv)';
                Y12 = grad_clipping(Y12,max_grad);

                

                [gradLB_mu,gradLB_Sig] = nat_gradients(setting,Y12,mu,Sig,Sig_inv);

                zeta = obj.parallel_transport_spd(Sig_inv_old, Sig, gradLB_Sig_momentum);
                
                gradLB_Sig_momentum = momentum_weight*zeta+(1-momentum_weight)*gradLB_Sig;
                gradLB_mu_momentum  = momentum_weight*gradLB_mu_momentum+(1-momentum_weight)*gradLB_mu;

                % Lower bound
                LB(iter) = mean(lb_log_h);

                % Smooth the lowerbound and store best results
                if iter>window_size
                    LB_smooth(iter-window_size) = mean(LB(iter-window_size:iter));
                    if LB_smooth(iter-window_size)>=max(LB_smooth(1:iter-window_size))
                        mu_best  = mu;
                        Sig_best = Sig;
                        patience = 0;
                    else
                        patience = patience + 1;
                    end
                end

                if (patience>max_patience)||(iter>max_iter)
                    stop = true;
                end

                % Display training information
                print_training_info(verbose,stop,iter,window_size,LB,LB_smooth)

                % If users want to save variational mean, var-cov matrix and ll, log_q at each iteration
                if(save_params)
                    ssave = write_iter_struct(ssave,save_params,iter,mu,llh_s,log_q_lambda_s,Sig_inv);
                end

            end

            % Store output

            LB_smooth = LB_smooth(1:(iter-window_size-1));
            LB = LB(1:iter-1);

            % Store output
            Post.LB0        = LB0;
            Post.LB         = LB;
            Post.LB_smooth  = LB_smooth;
            [Post.LB_max,Post.LB_indx] = max(LB_smooth);
            Post.mu         = mu_best;
            [~,Post.ll]     = hfunc(data,Post.mu,setting); %ll computed in posterior mean
            Post.Sig        = Sig_best;
            if isDiag
                Post.Sig2   = Post.Sig;
            else
                Post.Sig2   = diag(Post.Sig);
            end
            Post.setting = setting;
            if (save_params)
                Post.iter = ssave;
            end

            % Plot lowerbound
            if(lb_plot)
                obj.plot_lb(LB_smooth);
            end
        end

        %%

        function zeta = parallel_transport_spd(obj, X_inv, Y, eta)

            isDiag = isvector(X_inv);

            if isDiag
                E    = (Y.*X_inv).^0.5;
                zeta = E.^2.*eta;
            else
                E    = sqrtm((Y*X_inv));
                zeta = E*eta*E';
            end
        end


        function Y = retraction_spd(obj, X, X_inv, grad, step)

            isDiag = isvector(X);

            if isDiag
                teta      = step*grad;
                Y         = X + teta + .5*teta.*(teta.*X_inv);
            else
                teta      = step*grad;
                symm      = @(X) .5*(X+X');
                Y         = symm(X + teta + .5*teta*(X_inv*teta));
                [~,index] = chol(Y);
                iter      = 1;
                max_iter  = 5;
                while (index)&&(iter<=max_iter)
                    iter      = iter+1;
                    step      = step/2;
                    teta      = step*grad;
                    Y         = symm(X + teta + .5*teta*(X_inv*teta));
                    [~,index] = chol(Y);
                end
                if iter >= max_iter
                    Y = X;
                end
            end
        end
    end
end

function[ssave] = create_iter_struct(save_params,max_iter,setting)

ssave = struct();
d_theta = setting.d_theta;

if(save_params)
    ssave.mu    = zeros(max_iter,d_theta);
    ssave.ll    = zeros(max_iter,1);
    ssave.logq = zeros(max_iter,1);
    if ~setting.isDiag
        ssave.SigInv = zeros(max_iter,d_theta*d_theta);
    else
        ssave.SigInv = zeros(max_iter,d_theta);
    end
end
end

function[ssave] = write_iter_struct(ssave,save_params,iter,mu,llh_s,log_q_lambda_s,Sig_inv)

if(save_params)
    ssave.mu(iter,:)     = mu;
    ssave.ll(iter,:)     = mean(llh_s);
    ssave.logq(iter,:)   = mean(log_q_lambda_s);
    ssave.SigInv(iter,:) = Sig_inv(:)';
end
end

function Y12 = grad_clipping(Y12,max_grad)
if max_grad>0
    grad_norm = norm(Y12);
    norm_gradient_threshold = max_grad;
    if grad_norm > norm_gradient_threshold
        Y12 = (norm_gradient_threshold/grad_norm)*Y12;
    end
end
end

function[c] = control_variates(A,B,S,do)
if do
    c = (mean(A.*B)-mean(A).*mean(B))./var(B)*S/(S-1);
else
    c = 0;
end
end

function[] = print_training_info(verbose,stop,iter,window_size,LB,LB_smooth)
if verbose == 1
    if iter> window_size

        LBimporved = LB_smooth(iter-window_size)>=max(LB_smooth(1:iter-window_size));

        if LBimporved
            str = '*';
        else
            str = ' ';
        end

        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB_smooth(iter-window_size)),str])
    else
        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB(iter))])
    end
end

if verbose == 2 && stop == true
    if iter> window_size
        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB_smooth(iter-window_size))])
    else
        disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB(iter))])
    end
end
end


function[gradLB_mu,gradLB_iSig] = nat_gradients(setting,Y12,mu,Sig,Sig_inv)

isDiag      = setting.isDiag;
useHfunc    = setting.useHfunc;
Sig0_type   = setting.Sig0_type;
d_theta     = setting.d_theta;
mu0         = setting.mu0;
Sig_inv_0   = setting.Sig_inv_0;

if isDiag
    if useHfunc
        C_S     = 0;
        C_mu    = 0;
    else
        if Sig0_type == 1
            C_S     = -0.5*(diag(Sig_inv_0) - Sig_inv);
            C_mu    = -1./Sig_inv.*Sig_inv_0*(mu-mu0);
        else
            C_S     = -0.5*(Sig_inv_0-Sig_inv);
            C_mu    = -1./Sig_inv.*Sig_inv_0.*(mu-mu0);
        end
    end
    gradLB_mu       = C_mu + Y12(1:d_theta);
    gradLB_iSig     = Sig.*(C_S + Y12(d_theta+1:end)).*Sig;

else
    if useHfunc
        C_s     = 0;
        C_mu    = 0;
    else
        if Sig0_type == 1 %matrix
            C_s     = -0.5*(Sig_inv_0 -Sig_inv);
            C_mu    = -Sig*Sig_inv_0*(mu-mu0);
        elseif Sig0_type == 2 %vector
            C_s     = -0.5*(diag(Sig_inv_0) -Sig_inv);
            C_mu    = -Sig*(Sig_inv_0.*(mu-mu0));
        else %scalar
            C_s     = -0.5*(eye(d_theta)*Sig_inv_0 -Sig_inv);
            C_mu    = -Sig*(Sig_inv_0.*(mu-mu0));
        end
    end
    gradLB_mu       = C_mu + Y12(1:d_theta);
    gradLB_iSig     = Sig*(C_s  + reshape(Y12(d_theta+1:end),d_theta,d_theta))*Sig;

end

gradLB_iSig = gradLB_iSig*setting.SigFactor;

end

function[L,iS] = get_L_Sig(S,setting)
if ~setting.isDiag
    if setting.UseInvChol == 0
        iS = inv(S);
        L = chol(S,'lower');
    elseif setting.UseInvChol == 1
        L = chol(S,'lower');
        iL = inv(L);
        iS = iL'*iL;
    end
else
    iS = 1./S;
    L = S.^0.5;
end

end

function[setting, isDiag, useHfunc,UseInvChol] = check_setting(setting)


if ~isfield(setting,'UseInvChol')
    setting.UseInvChol = 0;
else
    if ~(setting.UseInvChol == 0 || setting.UseInvChol == 1)
        error('Invalid setting.UseInvChol field in setting.')
    end
end

if ~isfield(setting,'doCV')
    setting.doCV = 1;
else
    if ~(setting.doCV == -1 || setting.doCV == 0)
        error('Invalid setting.doCV field in setting.')
    end
end

if ~isfield(setting,'isDiag')
    error('Missing isDiag field in setting.')
end

if  ~isfield(setting,'useHfunc')
    error('Missing useHfunc field in setting.')
end

useHfunc    = setting.useHfunc;
isDiag      = setting.isDiag;
UseInvChol  = setting.UseInvChol;

[Sig_inv_0,Sig0_type] = prior_covariance(setting);

setting.Sig_inv_0 = Sig_inv_0;
setting.Sig0_type = Sig0_type;
if size(setting.Prior.Mu,2)>size(setting.Prior.Mu,1)
    setting.Prior.Mu = setting.Prior.Mu';
end
setting.mu0 = setting.Prior.Mu;

if numel(setting.mu0)>1 && numel(setting.mu0)~= setting.d_theta
    error('Size of parameters in Prior.Mu does not match the number of parameters.')
end

end

function[Sig_inv_0,Sig0_type] = prior_covariance(setting)

d_theta = setting.d_theta;

N_Sig_0 = numel(setting.Prior.Sig);
if N_Sig_0 == d_theta^2
    Sig0_type = 1; %matrix
    Sig_inv_0 = inv(setting.Prior.Sig);
else
    Sig_inv_0 = 1./setting.Prior.Sig;
    if N_Sig_0 >1
        Sig0_type = 2; % vector
    else
        Sig0_type = 3; %scalar
    end
end


if Sig0_type == 2 && numel(setting.Prior.Sig) ~= d_theta
    error('Number of elements in Prior.Sig (vector) do not match the number of parameters.')
end

if Sig0_type == 1 && numel(setting.Prior.Sig) ~= d_theta^2
    error('Number of elements in Prior.Sig (matrix) do not match the number of parameters.')
end

end

function[setting, SigFactor] = Factor_settings(setting)

if ~isfield(setting,'SigFactor')
    setting.SigFactor = 1;
else
    if setting.SigFactor ==0
        error('Invalid setting.SigFactor.')
    end
end
SigFactor = setting.SigFactor;
end