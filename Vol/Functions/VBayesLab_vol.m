classdef VBayesLab_vol
    %MODEL Abstract classes for models

    properties
        Method             % VB method -> a name
        Model              % Instance of the model to sample from
        ModelToFit         % Name of model to be fitted
        NumSample          % Number of samples to estimate the likelihood and gradient of likelihood
        GradWeight         % Momentum weight
        LearningRate       % Learning rate decay factor
        MaxIter            % Maximum number of VB iterations
        MaxPatience        % Maximum number of patiences for early stopping
        WindowSize         % Smoothing window
        ParamsInit         % Initial values of model parameters
        NumParams          % Number of model parameters
        Seed               % Random seed
        Post               % Struct to store estimation results
        Verbose            % Turn on of off printed message during sampling phase
        StdForInit         % Std of the normal distribution to initialize VB params
        MeanInit           % Pre-specified values of mean(theta)
        SigInitScale       % A constant to scale up or down std of normal distribution
        StepAdaptive       % From this iteration, stepsize is reduced
        LBPlot             % If user wants to plot the lowerbound at the end
        GradClipInit       % Apply gradient clipping from the very beginning
        GradientMax        % For gradient clipping
        InitMethod         % Method to initialize mu (variational mean)
        AutoDiff           % Turn on/off automatic differentiation
        HFunction          % Instance of function to compute h(theta)
        GradHFuntion       % Instance of function to compute gradient of h(theta)
        DataTrain          % Training data
        Setting            % Struct to store additional setting to the model
        SaveParams         % If save parameters in all iterations or not
        Optimization       % Optimization method
        P                  % Arch lags
        O                  % Asymmetry order
        Q                  % Garch lags
        GarchType          % garch / egarch / figarch /rgarch
        Sampler            % Solob or mvnrnd
        TestSet            % 1 if use training and test set, 0 if use all data
        Optimizer          % QBVI or MGVB or EMGVB
        doCV               % Use control variates
        useHfunc           % Use h function or gradient solutions
        SavePerf
    end

    methods


        %MODEL Construct an instance of this class
        function obj = VBayesLab_vol(varargin)

            obj.AutoDiff     = false;
            obj.GradientMax  = 100;
            obj.GradWeight   = 0.4;
            obj.LBPlot       = false;
            obj.LearningRate = 0.001;
            obj.MaxIter      = 5000;
            obj.MaxPatience  = 20;
            obj.NumSample    = 50;
            obj.StdForInit   = 0.01;
            obj.SigInitScale = 0.1;
            obj.StepAdaptive = floor(obj.MaxIter*0.80);
            obj.SaveParams   = false;
            obj.Verbose      = 2;
            obj.WindowSize   = 30;
            obj.GradClipInit = 0;
            obj.Sampler      = 'solob';
            obj.TestSet      = true;
            obj.doCV         = true;
            obj.useHfunc     = true;
            obj.SavePerf     = true;

        end


        % Plot lowerbound
        function plot_lb(obj,lb)
            plot(lb,'LineWidth',2)
            if(~isempty(obj.Model))
                title(['Lower bound ',obj.Method ,' - ',obj.Model.ModelName])
            else
                title('Lower bound')
            end
            xlabel('Iterations')
            ylabel('Lower bound')
        end


        % Sampler
        function[theta_all] = sampler(obj,mu,Sig)

            S = obj.NumSample;
            d_theta = obj.NumParams;
            sampler = obj.Sampler;

            switch sampler
                case 'solob'
                    rqmc        = utils_normrnd_qmc(S,d_theta);
                    C_lower     = chol(Sig,'lower');
                    theta_all   = mu +  C_lower*rqmc';
                case 'mvn'
                    theta_all   = mvnrnd(mu,Sig,S)';
                case 'n'
                    rqmc        = randn(S,d_theta);
                    C_lower     = chol(Sig,'lower');
                    theta_all   = mu +  C_lower*rqmc';
            end
        end



        % Optimizer: main update
        function[mu,Sig,Sig_inv,Sig_old] = update(obj,stepsize,mu,Sig,Sig_inv,gradLB_Sig_momentum,gradLB_mu_momentum)

            switch obj.Optimizer
                case 'QBVI'
                    if obj.useHfunc
                        Sig_inv = Sig_inv + stepsize*gradLB_Sig_momentum;
                    else
                        Sig_inv = (1-stepsize)*Sig_inv + stepsize*gradLB_Sig_momentum;
                    end
                    Sig         = inv(Sig_inv);
                    mu          = mu + stepsize*Sig*gradLB_mu_momentum;
                    Sig_old     = nan;

                case 'MGVB'
                    Sig_old     = Sig_inv;
                    mu          = mu + stepsize*gradLB_mu_momentum;
                    Sig         = retraction_spd(Sig,Sig_inv,gradLB_Sig_momentum,stepsize);
                    Sig_inv     = inv(Sig);

                case 'EMGVB'
                    Sig_old     = Sig;
                    mu          = mu + stepsize*gradLB_mu_momentum;
                    Sig_inv     = retraction_spd(Sig_inv,Sig,gradLB_Sig_momentum, stepsize);
                    Sig         = inv(Sig_inv);

                case 'BBVI'
                    mu          = mu + stepsize*gradLB_mu_momentum;
                    Sig         = Sig + stepsize*gradLB_Sig_momentum;
                    Sig_inv     = inv(Sig);
                    Sig_old     = nan;
            end
        end



        % Optimizer: gradient of logq wrt Sigma
        function[grad] = fun_gra_log_q_Sig(obj,Sig_inv,aux)
            switch obj.Optimizer
                case 'QBVI'
                    grad = Sig_inv-Sig_inv*aux*(aux')*Sig_inv;
                case 'MGVB'
                    grad = -1/2*Sig_inv+1/2*Sig_inv*aux*(aux')*Sig_inv;
                case 'EMGVB'
                    grad = Sig_inv-Sig_inv*aux*(aux')*Sig_inv;
                case 'BBVI'
                    grad = -1/2*Sig_inv+1/2*Sig_inv*aux*(aux')*Sig_inv;
            end
        end


        % Optimizer: LB gradient and momentum
        function[gradLB_mu_momentum,gradLB_Sig_momentum] = LBgradients(obj,Y12,d_theta,gradLB_Sig_momentum,gradLB_mu_momentum,Sig,Sig_old,Sig_inv,iS0,mu0,mu)

            switch obj.Optimizer
                case 'QBVI'
                    if obj.useHfunc
                        gradLB_mu   = Sig_inv*Y12(1:d_theta);
                        gradLB_Sig  = reshape(Y12(d_theta+1:end),d_theta,d_theta);
                    else
                        gradLB_mu   = iS0*(mu0-mu) + Sig_inv*Y12(1:d_theta);
                        gradLB_Sig  = iS0 + reshape(Y12(d_theta+1:end),d_theta,d_theta);
                    end
                case 'MGVB'
                    if obj.useHfunc
                        gradLB_mu   = Y12(1:d_theta);
                        gradLB_Sig  = 2*Sig*reshape(Y12(d_theta+1:end),d_theta,d_theta)*Sig;
                    else
                        gradLB_mu   = -Sig*iS0*(mu-mu0) + Y12(1:d_theta);
                        gradLB_Sig  = 2*Sig*(-0.5*(iS0-Sig_inv) + reshape(Y12(d_theta+1:end),d_theta,d_theta))*Sig;
                    end
                case 'EMGVB'
                    if obj.useHfunc
                        gradLB_mu   = Y12(1:d_theta);
                        gradLB_Sig  = reshape(Y12(d_theta+1:end),d_theta,d_theta);
                    else
                        gradLB_mu = -Sig*iS0*(mu-mu0) + Y12(1:d_theta);
                        gradLB_Sig = iS0-Sig_inv + reshape(Y12(d_theta+1:end),d_theta,d_theta);
                    end
                case 'BBVI'
                    if obj.useHfunc
                        gradLB_mu   = Sig_inv*Y12(1:d_theta);
                        gradLB_Sig  = reshape(Y12(d_theta+1:end),d_theta,d_theta);
                    else
                        gradLB_mu   = -Sig*iS0*(mu-mu0) + Y12(1:d_theta);
                        gradLB_Sig  = -0.5*(iS0-Sig_inv) + reshape(Y12(d_theta+1:end),d_theta,d_theta);
                    end                    
            end


            if isempty(gradLB_Sig_momentum) % Used for initialization
                gradLB_Sig_momentum = gradLB_Sig;
                gradLB_mu_momentum  = gradLB_mu;
            else
                momentum_weight = obj.GradWeight;
                switch obj.Optimizer
                    case {'QBVI','BBVI'}
                        zeta = gradLB_Sig_momentum;
                    case 'MGVB'
                        zeta = parallel_transport_spd(Sig_old,Sig,gradLB_Sig_momentum);
                    case 'EMGVB'
                        zeta = parallel_transport_spd(Sig_old,Sig_inv,gradLB_Sig_momentum);
                end
                gradLB_Sig_momentum = momentum_weight*zeta+(1-momentum_weight)*gradLB_Sig;
                gradLB_mu_momentum  = momentum_weight*gradLB_mu_momentum+(1-momentum_weight)*gradLB_mu;
            end
        end


        % Print info
        function[] = print_training_info(obj,stop,iter,LB,LB_smooth)
            if obj.Verbose == 1
                if iter>obj.WindowSize

                    LBimporved = LB_smooth(iter)>=max(LB_smooth(1:iter),[],'omitnan');

                    if LBimporved
                        str = '*';
                    else
                        str = ' ';
                    end

                    disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB_smooth(iter)),str])
                else
                    disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB(iter))])
                end
            end

            if obj.Verbose == 2 && stop == true
                if iter> obj.WindowSize
                    disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB_smooth(iter))])
                else
                    disp(['Iter: ',num2str(iter,'%04i'),'| LB: ',num2str(LB(iter))])
                end
            end
        end

        function Y12 = grad_clipping(obj,Y12,init)
            if init
                max_grad = obj.GradClipInit;
            else
                max_grad = obj.GradientMax;
            end

            if max_grad>0
                grad_norm = norm(Y12);
                norm_gradient_threshold = max_grad;
                if grad_norm > norm_gradient_threshold
                    Y12 = (norm_gradient_threshold/grad_norm)*Y12;
                end
            end
        end


        function[c] = control_variates(obj,A,B)
            S = obj.NumSample;
            if obj.doCV
                c = (mean(A.*B)-mean(A).*mean(B))./var(B)*S/(S-1);
            else
                c = 0;
            end
        end



    end

end


function zeta = parallel_transport_spd(X_inv, Y, eta)

isDiag = isvector(X_inv);

if isDiag
    E    = (Y.*X_inv).^0.5;
    zeta = E.^2.*eta;
else
    E    = sqrtm((Y*X_inv));
    zeta = E*eta*E';
end
end


function Y = retraction_spd(X, X_inv, grad, step)

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

