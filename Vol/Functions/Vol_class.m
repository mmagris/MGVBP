classdef Vol_class < VBayesLab_vol

    properties

    end

    methods
        function obj = Vol_class(data,varargin)

            obj.Method = 'Vol_class';

            % Parse additional options
            if nargin > 2
                paramNames = ...
                    {'NumSample'        'LearningRate'      'GradWeight'       'GradClipInit' ...
                    'MaxIter'           'MaxPatience'       'WindowSize'       'Verbose' ...
                    'InitMethod'        'StdForInit'        'Seed'             'MeanInit' ...
                    'SigInitScale'      'LBPlot'            'GradientMax'      'DataTrain' ...
                    'Setting'           'StepAdaptive'      'SaveParams'...
                    'P'                 'O'                 'Q'                'GarchType'...
                    'Sampler'           'TestSet'           'Optimizer'        'doCV'...
                    'useHfunc'          'SavePerf'};

                paramDflts = ...
                    {obj.NumSample      obj.LearningRate    obj.GradWeight    obj.GradClipInit ...
                    obj.MaxIter         obj.MaxPatience     obj.WindowSize    obj.Verbose ...
                    obj.InitMethod      obj.StdForInit      obj.Seed          obj.MeanInit ...
                    obj.SigInitScale    obj.LBPlot          obj.GradientMax   obj.DataTrain ...
                    obj.Setting         obj.StepAdaptive    obj.SaveParams ...
                    obj.P               obj.O               obj.Q             obj.GarchType...
                    obj.Sampler         obj.TestSet       obj.Optimizer     obj.doCV...
                    obj.useHfunc        obj.SavePerf};

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
                    obj.DataTrain,...
                    obj.Setting,...
                    obj.StepAdaptive,...
                    obj.SaveParams,...
                    obj.P,...
                    obj.O,...
                    obj.Q,...
                    obj.GarchType,...
                    obj.Sampler,...
                    obj.TestSet,...
                    obj.Optimizer,...
                    obj.doCV,...
                    obj.useHfunc,...
                    obj.SavePerf] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});

            end

            switch obj.GarchType
                case {'garch','egarch'}
                    obj.NumParams = obj.P+obj.O+obj.Q+1;
                case 'figarch'
                    obj.NumParams = obj.P+obj.Q+2;
                case 'rgarch'
                    obj.NumParams = obj.P+obj.Q+7;
            end

            obj.HFunction = h_fun;

            if isempty(obj.MeanInit)
                mu = normrnd(0,obj.StdForInit,obj.NumParams,1);
                obj.InitMethod = 'Random';
                obj.MeanInit = mu;
            elseif obj.MeanInit == 0
                obj.InitMethod = 'Zeros';
                obj.MeanInit = zeros(obj.NumParams,1);
            else
                if size(obj.MeanInit,1) ~= obj.NumParams
                    error(['Initial mu must be of dimension (', num2str(obj.NumParams,'%i') 'x1)!'])
                end
                obj.InitMethod = 'Fixed';
            end


            % Main function to run QBVI
            obj.Post   = obj.fit(data);
        end

        function[tpar] = itransform(obj,par)
            tpar = garch_itransform(par,obj.P,obj.O,obj.Q,obj.GarchType);
        end

        function[SigSymm] = Symmetrize(obj)
            if ~all(all(obj.Post.Sig == obj.Post.Sig')) % not symmetric posterior
                warning([obj.Optimizer '.Post.Sig is not symmetric, so it is being symmetrized.'])
            end
            SigSymm = 1/2*(obj.Post.Sig + obj.Post.Sig');
        end

        function[dis] = get_xy_distribution(obj,Num_xp,k,trans)
            NumParams   = obj.NumParams;
            x        = zeros(Num_xp,NumParams);
            y        = zeros(Num_xp,NumParams);
            if ~trans

                %  Use kernel distr on posterior samples
                % c = mvnrnd(obj.Post.mu,obj.Post.Sig,80000);
                % rmea    = mean(c);
                % rstd    = std(c);
                % for i = 1:NumParams
                %x(:,i) = linspace(rmea(i)-k*rstd(i),rmea(i)+k*rstd(i),Num_xp);
                %y(:,i) = ksdensity(c(:,i),x(:,i));
                % end

                % Use posterior normal pdf
                pmea = obj.Post.mu;
                pstd = sqrt(obj.Post.Sig2);

                for i = 1:NumParams
                    x(:,i) = linspace(pmea(i)-k*pstd(i),pmea(i)+k*pstd(i),Num_xp);
                    y(:,i) = normpdf(x(:,i),pmea(i),pstd(i));
                end

            else
                SymmSig = Symmetrize(obj);
                N_draws = 80000;
                r = mvnrnd(obj.Post.mu,SymmSig,N_draws);
                c = zeros(N_draws,NumParams);

                for i = 1:N_draws
                    c(i,:) =  obj.itransform(r(i,:)');
                end

                rmea = mean(c);
                rstd = std(c);

                for i = 1:NumParams
                    x(:,i) = linspace(rmea(i)-k*rstd(i),rmea(i)+k*rstd(i),Num_xp);
                    y(:,i) = ksdensity(c(:,i),x(:,i));
                end

            end
            dis.x = x;
            dis.y = y;
            dis.k = k;
            dis.Num_xp = Num_xp;
            dis.transform = logical(trans);
        end

        %% main function

        function Post = fit(obj,data)

            d_theta = obj.NumParams;

            % Extract sampling setting

            eps0            = obj.LearningRate;
            S               = obj.NumSample;
            window_size     = obj.WindowSize;
            init_scale      = obj.SigInitScale;
            stepsize_adapt  = obj.StepAdaptive;
            hfunc           = obj.HFunction;
            setting         = obj.Setting;

            if obj.TestSet
                if isempty(data.test)
                    error('Empty test data.')
                end
            else
                data.train = data.all;
            end

            % Initialization
            iter      = 0;
            patience  = 0;
            stop      = false;
            LB              = zeros(1,obj.MaxIter);
            LB_test         = zeros(1,obj.MaxIter);
            LB_smooth       = nan(1,obj.MaxIter);
            LB_smooth_test  = nan(1,obj.MaxIter);

            S0          = setting.Prior(2).*eye(d_theta);
            iS0         = inv(S0);
            mu0         = setting.Prior(1).*ones(d_theta,1);


            % Initialization of mu
            mu = obj.MeanInit;

            % Initialization of Sig
            Sig     = init_scale*eye(d_theta);
            Sig_inv = inv(Sig);

            log_pdf = @(theta,mu,Sig_inv) -d_theta/2*log(2*pi)+1/2*log(det(Sig_inv))-1/2*(theta-mu)'*Sig_inv*(theta-mu);

            nPar                        = d_theta+d_theta*d_theta;
            gra_log_q_lambda            = zeros(S,nPar);
            grad_log_q_h_function       = zeros(S,nPar);
            grad_log_q_h_function_cv    = zeros(S,nPar);
            c12                         = zeros(1,nPar);
            lb_log_h                    = zeros(S,1);

            tr_mat_s = zeros(S,4);
            if obj.TestSet
                te_mat_s = zeros(S,4);
            end

            if obj.SavePerf
                Perf.train      = zeros(obj.MaxIter,7);
                Perf.test       = zeros(obj.MaxIter,7);
            end

            theta_all = obj.sampler(mu,Sig);

            for s = 1:S
                % Parameters from Normal distribution
                theta = theta_all(:,s);

                % Log q_lambda
                log_q_lambda = log_pdf(theta,mu,Sig_inv);

                % h function components
                [h_theta,llh,pri] = hfunc(data.train(:,1),theta,obj);

                % h function
                h_function = h_theta - log_q_lambda;

                if obj.useHfunc
                    f = h_function;
                else
                    f = llh;
                end


                % Compute the LB
                lb_log_h(s) = h_function;

                tr_mat_s(s,:) = [llh,pri,log_q_lambda,h_function];
                if obj.TestSet
                    [h_theta2,llh2,pri2]  = hfunc(data.test,theta,obj);
                    h_function2 = h_theta2 - log_q_lambda;
                    te_mat_s(s,:) = [llh2,pri2,log_q_lambda,h_function2];
                end

                aux                           = (theta-mu);
                gra_log_q_mu                  = aux;
                gra_log_q_Sig                 = obj.fun_gra_log_q_Sig(Sig_inv,aux);
                gra_log_q_lambda(s,:)         = [gra_log_q_mu;gra_log_q_Sig(:)]';
                grad_log_q_h_function(s,:)    = gra_log_q_lambda(s,:)*f;
                grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(f-c12);
            end

            c12 = obj.control_variates(grad_log_q_h_function,gra_log_q_lambda);
            Y12 = mean(grad_log_q_h_function_cv)';

            % Gradient clipping at the beginning
            Y12 = obj.grad_clipping(Y12,1);

            [gradLB_mu_momentum,gradLB_Sig_momentum] = obj.LBgradients(Y12,d_theta,[],[],Sig,[],Sig_inv,iS0,mu0,mu);

            % Compute LB0
            LB0 = mean(lb_log_h);

            % Matrices 0
            it.te_mat_0 = mean(tr_mat_s,1);
            if obj.TestSet
                it.tr_mat_0 = mean(te_mat_s,1);
            end

            % Performance 0
            if obj.SavePerf
                Perf.train_0 = garch_perf(obj,data.train,mu);
                if obj.TestSet
                    Perf.test_0 = garch_perf(obj,data.train,mu);
                end
            end

            % Save params 0
            it.mu_0  = mu;
            it.S_0   = Sig(:)';
            it.par_0 = obj.itransform(mu);


            if obj.Verbose ~=0
                disp(['Iter: 0000 |LB: ', num2str(LB0)])
            end

            % Prepare for the next iterations
            mu_best     = mu;
            Sig_best    = Sig;


            it.tr_mat = zeros(obj.MaxIter,4);
            if obj.TestSet
                it.te_mat = zeros(obj.MaxIter,4);
            end

            if obj.SaveParams

            end

            while ~stop

                iter = iter+1;
                if iter>stepsize_adapt
                    stepsize = eps0*stepsize_adapt/iter;
                else
                    stepsize = eps0;
                end

                [mu,Sig,Sig_inv,Sig_old] = obj.update(stepsize,mu,Sig,Sig_inv,gradLB_Sig_momentum,gradLB_mu_momentum);

                gra_log_q_lambda            = zeros(S,nPar);
                grad_log_q_h_function       = zeros(S,nPar);
                grad_log_q_h_function_cv    = zeros(S,nPar);

                theta_all = obj.sampler(mu,Sig);

                for s = 1:S
                    % Parameters from Normal distribution
                    theta = theta_all(:,s);

                    % Log q_lambda
                    log_q_lambda = log_pdf(theta,mu,Sig_inv);

                    % h function
                    [h_theta,llh,pri] = hfunc(data.train(:,1),theta,obj);

                    % h function
                    h_function = h_theta - log_q_lambda;

                    if obj.useHfunc
                        f = h_function;
                    else
                        f = llh;
                    end

                    % Compute the lowerbound
                    lb_log_h(s) = h_function;

                    tr_mat_s(s,:) = [llh,pri,log_q_lambda,h_function];
                    if obj.TestSet
                        [h_theta2,llh2,pri2]  = hfunc(data.test,theta,obj);
                        h_function2 = h_theta2 - log_q_lambda;
                        te_mat_s(s,:) = [llh2,pri2,log_q_lambda,h_function2];
                    end

                    aux                           = (theta-mu);
                    gra_log_q_mu                  = aux;
                    gra_log_q_Sig                 = obj.fun_gra_log_q_Sig(Sig_inv,aux);
                    gra_log_q_lambda(s,:)         = [gra_log_q_mu;gra_log_q_Sig(:)]';
                    grad_log_q_h_function(s,:)    = gra_log_q_lambda(s,:)*f;
                    grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(f-c12);

                end

                c12 = obj.control_variates(grad_log_q_h_function,gra_log_q_lambda);
                Y12 = mean(grad_log_q_h_function_cv)';

                % Clipping the gradient
                Y12 = obj.grad_clipping(Y12,0);

                [gradLB_mu_momentum,gradLB_Sig_momentum] = obj.LBgradients(Y12,d_theta,gradLB_Sig_momentum,gradLB_mu_momentum,Sig,Sig_old,Sig_inv,iS0,mu0,mu);

                % Lower bound
                LB(iter) = mean(lb_log_h);

                % Matrices
                it.tr_mat(iter,:) = mean(tr_mat_s,1);
                if obj.TestSet
                    it.te_mat(iter,:) = mean(te_mat_s,1);
                    LB_test(iter) = it.te_mat(iter,4);
                end

                % Performance
                if obj.SavePerf
                    Perf.train(iter,:) = garch_perf(obj,data.train,mu);
                    if obj.TestSet
                        Perf.test(iter,:)  = garch_perf(obj,data.test,mu);
                    end
                end

                % Save params at each iteration
                if obj.SaveParams
                    it.mu(iter,:)       = mu;
                    it.S(iter,:)        = Sig(:)';
                    it.par(iter,:)      = obj.itransform(mu);
                end

                % Smooth the lowerbound and store best results
                if iter>window_size
                    LB_smooth(iter) = mean(LB(iter-window_size:iter));
                    if obj.TestSet
                        LB_smooth_test(iter) = mean(LB_test(iter-window_size:iter));
                    end

                    if LB_smooth(iter)>=max(LB_smooth(1:iter),[],'omitnan')
                        mu_best  = mu;
                        Sig_best = Sig;
                        patience = 0;
                        iter_best = iter;
                    else
                        patience = patience + 1;
                    end

                end

                if (patience>obj.MaxPatience)||(iter>=obj.MaxIter)
                    stop = true;
                end

                % Display training information
                obj.print_training_info(stop,iter,LB,LB_smooth)

            end

            % Store output
            Post.LB0                = LB0;
            Post.LB                 = LB;
            Post.LB_smooth          = LB_smooth;
            [Post.LB_max,Post.LB_indx] = max(LB_smooth);
            
            if iter_best ~= Post.LB_indx
                warning('Check: something strange.')
            end

            Post.iter_best          = iter_best;
            Post.mu                 = mu_best;
            Post.tpar               = Post.mu;
            Post.par                = obj.itransform(mu_best);
            Post.Sig                = Sig_best;
            Post.Sig2               = diag(Post.Sig);
           

            % Stats in Variational Posterior Mean
            [h_theta3,llh3,pri3]    = hfunc(data.train(:,1),Post.mu,obj);   
            log_q3                  = log_pdf(Post.mu,Post.mu,inv(Post.Sig));
            [~,Post.ht.train]       = garch_nll_ht(data.train(:,1),Post.mu,obj);  
            Post.Stat.train         = [llh3,pri3,log_q3,h_theta3-log_q3];
            % Mean Stats for the Variational Posterior
            Post.Stat.E_train       = it.tr_mat(iter_best,:);
            Post.Stat.iter          = it;

            if obj.TestSet
                [h_theta3,llh3,pri3]    = hfunc(data.test(:,1),Post.mu,obj);
                log_q3                  = log_pdf(Post.mu,Post.mu,inv(Post.Sig));
                [~,Post.ht.test]        = garch_nll_ht(data.test(:,1),Post.mu,obj);
                Post.Stat.test          = [llh3,pri3,log_q3,h_theta3-log_q3];
                Post.Stat.E_test        = it.te_mat(iter_best,:); %expected values of e.g. the llh

                Post.LB_test.LB = LB_test;
                Post.LB_test.LB_smooth = LB_smooth_test;
                Post.LB_test.LB_max = LB_smooth_test(iter_best);
            end

            if obj.SavePerf                
                Post.Perf.train     = garch_perf(obj,data.train,Post.mu);                
                if obj.TestSet
                    Post.Perf.test  = garch_perf(obj,data.test,Post.mu);                    
                end
                Post.Perf.iter      = Perf;
            end
               

            % Plot LB
            if(obj.LBPlot)
                obj.plot_lb(LB_smooth);
            end

            if obj.Verbose >0
                fprintf([obj.Optimizer ', Max. LB : %.4f.\n\n'],Post.LB_max)
            end
        end


    end
end

%% Functions definitions

function[fun] = h_fun

fun = @(data,theta,obj) grad_h_func(data,theta,obj);

    function [h_func,llh,log_prior] = grad_h_func(data,theta,obj)

        % Extract additional settings
        d = length(theta);
        sigma2 = obj.Setting.Prior(2);

        % Extract data
        y = data(:,end);

        % Compute log likelihood
        switch obj.GarchType
            case 'garch'
                llh = -fun_tarch_nll(y,theta,obj.P,obj.O,obj.Q,1);
            case 'egarch'
                llh = -fun_egarch_nll(y,theta,obj.P,obj.O,obj.Q,1);
            case 'figarch'
                llh = -fun_figarch_nll(y,theta,obj.P,obj.Q,0.5,1);
            case 'rgarch'
                llh = -fun_rgarch_nll(data,theta,obj.P,obj.Q,1);
        end

        % Compute log prior
        log_prior = -d/2*log(2*pi)-d/2*log(sigma2)-theta'*theta/sigma2/2;

        % Compute h(theta) = log p(y|theta) + log p(theta)
        h_func = llh + log_prior;

    end

end


% function[nll,ht] = garch_nll_ht(data,par,obj)
% 
% switch obj.GarchType
%     case 'garch'
%         [nll,ht]   = fun_tarch_nll(data,par,obj.P,obj.O,obj.Q,1);
%     case 'egarch'
%         [nll,ht]   = fun_egarch_nll(data,par,obj.P,obj.O,obj.Q,1);
%     case 'figarch'
%         [nll,ht]   = fun_figarch_nll(data,par,obj.P,obj.Q,0.5,1);
%     case 'rgarch'
%         [nll,ht]   = fun_rgarch_nll(data,par,obj.P,obj.Q,1);
% end
% 
% end
% 
% 
% 
% function[out] = garch_perf(obj,data,mu)
% 
% [nll,ht] = garch_nll_ht(data(:,1),mu,obj);
% mse     = @(col) mean((data(:,col).^2-ht).^2);
% qlik    = @(col) mean(data(:,col).^2./ht-log(data(:,col).^2./ht)-1);
% 
% out = [-nll,mse(1),mse(2),mse(3),qlik(1),qlik(2),qlik(3)];
% 
% end
