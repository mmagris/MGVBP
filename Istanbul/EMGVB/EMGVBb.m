classdef EMGVBb < VBayesLab
    %MVB Summary of this class goes here
    %   Detailed explanation goes here

    properties
        GradClipInit       % If doing gradient clipping at the beginning
    end

    methods
        function obj = EMGVBb(mdl,data,varargin)
            %MVB Construct an instance of this class
            %   Detailed explanation goes here
            obj.Method        = 'EMGVBb';
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
            setting = setting_block(setting);
            [setting, useHfunc, ~] = check_setting(setting);

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
                if size(mu,2)>size(mu,1)
                    mu = mu';
                end
            end

            B = setting.Block;
            if sum(B.blks) ~= d_theta
                error('Check block sizes.')
            end

            n_blks      = B.n;
            n_par_all   = sum(B.blks)+sum(B.blks.^2);
            Sig_inv     = cell(n_blks,1);

            for b = 1:n_blks
                indx        = B.blks(b);
                Sig_inv{b,1}    = 1/init_scale*eye(indx);
            end

            % Functions
            sim_theta   = @(s,mu,C_lower,rqmc) mu + C_lower*rqmc(s,:)';
            log_pdf     = @(theta,mu,Sig_inv,d_theta) -d_theta/2*log(2*pi)+1/2*log(det(Sig_inv))-1/2*(theta-mu)'*Sig_inv*(theta-mu);
            fun_gra_log_q_Sig = @(aux,Sig_inv) Sig_inv-Sig_inv*aux*(aux')*Sig_inv;



            aI = cell(n_blks,1);
            for b = 1:n_blks
                aI{b} = fliplr(eye(B.blks(b)));
            end
            mk_L_Sig = @(Sig_inv,aI) get_L_Sig(Sig_inv,setting.UseInvChol,aI);


            if setting.doCV == 1
                fun_cv = @(A,B) control_variates(A,B,S,1);
            else
                fun_cv = @(A,B) control_variates(A,B,S,0);
            end

            c12                         = zeros(1,n_par_all);
            gra_log_q_lambda            = zeros(S,n_par_all);
            grad_log_q_h_function       = zeros(S,n_par_all);
            grad_log_q_h_function_cv    = zeros(S,n_par_all);
            lb_log_h                    = zeros(S,1);
            L                           = cell(n_blks,1);
            Sig                         = cell(n_blks,1);
            gra_log_q_Sig               = zeros(sum(B.blks.^2),1);
            theta                       = zeros(d_theta,1);

            if save_params
                llh_s = zeros(S,1);
                log_q_lambda_s = zeros(S,1);
            end

            for b = 1:n_blks
                [L{b,1},Sig{b,1}] = mk_L_Sig(Sig_inv{b},aI{b});
            end

            rqmc = utils_normrnd_qmc(S,d_theta);

            for s = 1:S
                % Parameters in Normal distribution
                for b = 1:n_blks
                    indx = B.indx{b,1};
                    theta(indx,1) = sim_theta(s,mu(indx),L{b},rqmc(:,indx));
                end
                [h_theta,llh] = hfunc(data,theta,setting);

                % Log q_lambda
                log_q_lambda = 0;
                for b = 1:n_blks
                    indx = B.indx{b,1};
                    log_q_lambda = log_q_lambda + log_pdf(theta(indx),mu(indx),Sig_inv{b},B.blks(b));
                end

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

                aux                           = (theta-mu);
                gra_log_q_mu                  = aux;

                for b = 1:n_blks
                    indx    = B.indx{b,1};
                    indx_3  = B.indx{b,2};
                    tmp     = fun_gra_log_q_Sig(aux(indx),Sig_inv{b});
                    gra_log_q_Sig(indx_3,1) = tmp(:);
                end

                gra_log_q_lambda(s,:)         = [gra_log_q_mu;gra_log_q_Sig(:)]';
                grad_log_q_h_function(s,:)    = gra_log_q_lambda(s,:)*f;
                grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(f-c12);
            end

            c12 = fun_cv(grad_log_q_h_function,gra_log_q_lambda);
            Y12 = mean(grad_log_q_h_function_cv)';
            Y12 = grad_clipping(Y12,max_grad_init);

            [gradLB_mu,gradLB_iSig] = nat_gradients(setting,Y12,mu,Sig,Sig_inv);

            gradLB_iSig_momentum    = gradLB_iSig;
            gradLB_mu_momentum      = gradLB_mu;

            LB0 = mean(lb_log_h);
            if verbose ~=0
                disp(['Iter: 0000 |LB: ', num2str(LB0)])
            end

            % Prepare for the next iterations
            mu_best = mu;
            Sig_inv_best = Sig_inv;

            zeta = cell(B.n,1);

            while ~stop

                iter = iter+1;
                if iter>stepsize_adapt
                    stepsize = eps0*stepsize_adapt/iter;
                else
                    stepsize = eps0;
                end

                Sig_old     = Sig;
                mu          = mu + stepsize*gradLB_mu_momentum;
                for b = 1:setting.Block.n 
                    % EMGVB updates Sig_inv, MGVB updates Sig
                    Sig_inv{b,1}  = obj.retraction_spd(Sig_inv{b}, Sig{b}, gradLB_iSig_momentum{b}, stepsize);
                end

                for b = 1:setting.Block.n
                    % Sig needed in retraction and parallel transport
                    [L{b},Sig{b}] = mk_L_Sig(Sig_inv{b},aI{b});
                end

                if save_params
                    llh_s = zeros(S,1);
                    log_q_lambda_s = zeros(S,1);
                end

                rqmc     = utils_normrnd_qmc(S,d_theta);


                for s = 1:S
                    % Parameters in Normal distribution
                    for b = 1:n_blks
                        indx = B.indx{b,1};
                        theta(indx,1) = sim_theta(s,mu(indx),L{b},rqmc(:,indx));
                    end
                    [h_theta,llh] = hfunc(data,theta,setting);

                    % Log q_lambda
                    log_q_lambda = 0;
                    for b = 1:n_blks
                        indx = B.indx{b,1};
                        log_q_lambda = log_q_lambda + log_pdf(theta(indx),mu(indx),Sig_inv{b},B.blks(b));
                    end

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

                    aux                           = (theta-mu);
                    gra_log_q_mu                  = aux;

                    for b = 1:n_blks
                        indx    = B.indx{b,1};
                        indx_3  = B.indx{b,2};
                        tmp     = fun_gra_log_q_Sig(aux(indx),Sig_inv{b});
                        gra_log_q_Sig(indx_3,1) = tmp(:);
                    end

                    gra_log_q_lambda(s,:)         = [gra_log_q_mu;gra_log_q_Sig(:)]';
                    grad_log_q_h_function(s,:)    = gra_log_q_lambda(s,:)*f;
                    grad_log_q_h_function_cv(s,:) = gra_log_q_lambda(s,:).*(f-c12);
                end

                c12 = fun_cv(grad_log_q_h_function,gra_log_q_lambda);
                Y12 = mean(grad_log_q_h_function_cv)';
                Y12 = grad_clipping(Y12,max_grad);

                [gradLB_mu,gradLB_iSig] = nat_gradients(setting,Y12,mu,Sig,Sig_inv);

                for b = 1:n_blks
                    zeta{b} = obj.parallel_transport_spd(Sig_old{b}, Sig_inv{b}, gradLB_iSig_momentum{b});
                    gradLB_iSig_momentum{b} = momentum_weight*zeta{b}+(1-momentum_weight)*gradLB_iSig{b};
                end
                gradLB_mu_momentum   = momentum_weight*gradLB_mu_momentum+(1-momentum_weight)*gradLB_mu;

                % Lower bound
                LB(iter) = mean(lb_log_h);

                % Smooth the lowerbound and store best results
                if iter>window_size
                    LB_smooth(iter-window_size) = mean(LB(iter-window_size:iter));
                    if LB_smooth(iter-window_size)>=max(LB_smooth(1:iter-window_size))
                        mu_best  = mu;
                        Sig_inv_best = Sig_inv;
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
                    ssave = write_iter_struct_blk(ssave,save_params,iter,mu,llh_s,log_q_lambda_s,Sig_inv);
                end

            end




            LB_smooth = LB_smooth(1:(iter-window_size-1));
            LB = LB(1:iter-1);

            % Store output
            Post.LB0        = LB0;
            Post.LB         = LB;
            Post.LB_smooth  = LB_smooth;
            [Post.LB_max,Post.LB_indx] = max(LB_smooth);
            Post.mu         = mu_best;
            [~,Post.ll]     = hfunc(data,Post.mu,setting); %ll computed in posterior mean
            Post.Sig_inv    = blkdiag(Sig_inv_best{:});
            for b = 1:n_blks
                [~,Post.Sig{b}] = get_L_Sig(Sig_inv_best{b},1,aI{b});
            end
            Post.Sig   = blkdiag(Post.Sig{:});
            Post.Sig2   = diag(Post.Sig);

            Post.setting = setting;
            if(save_params)
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





