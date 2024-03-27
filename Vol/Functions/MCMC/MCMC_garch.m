classdef MCMC_garch < handle & matlab.mixin.CustomDisplay
    
    properties
        Method
        Model           % Instance of model to be fitted
        ModelToFit      % Name of model to be fitted
        SeriesLength    % Length of the series    
        NumMCMC         % Number of MCMC iterations 
        BurnInRate      % Percentage of sample for burnin
        BurnIn          % Number of samples for burnin
        TargetAccept    % Target acceptance rate
        NumCovariance   % Number of latest samples to calculate adaptive covariance matrix for random-walk proposal
        SaveFileName    % Save file name
        SaveAfter       % Save the current results after each 5000 iteration
        ParamsInit      % Initial values of model parameters
        Seed            % Random seed
        Post            % Struct to store estimation results
        Initialize      % Initialization method
        LogLikelihood   % Handle of the log-likelihood function
        PrintMessage    % Custom message during the sampling phase
        CPU             % Sampling time    
        Verbose         % Turn on of off printed message during sampling phase
        SigScale
        Scale           % Adaptive scale for proposal distribution
        Params          % To store MC means and std, after calling getParamsMean()
        Perf
    end
    
    methods
        function obj = MCMC_garch(model,data,varargin)
            % Constructs an instance of this class

            obj.Method        = 'MCMC';
            obj.Model         = model;   
            obj.ModelToFit    = model.ModelName;
            obj.NumMCMC       = 10000;
            obj.BurnInRate    = 0.2;
            obj.TargetAccept  = 0.25;            
            obj.NumCovariance = 2000;
            obj.SigScale      = 0.01;
            obj.Scale         = 1;
            obj.SaveAfter     = 0;
            obj.Verbose       = 1000;
            obj.ParamsInit    = model.ParamsInit;

            if nargin > 2
                %Parse additional options
                paramNames = {'NumMCMC'         'BurnInRate'      'TargetAccept'      'NumCovariance'   ...
                                                'SaveFileName'    'SaveAfter'         'Verbose'     ...
                              'Seed'            'SigScale'        'Scale'             
                              };

                paramDflts = {obj.NumMCMC       obj.BurnInRate    obj.TargetAccept    obj.NumCovariance  ...
                                                obj.SaveFileName  obj.SaveAfter       obj.Verbose   ...
                              obj.Seed          obj.SigScale      obj.Scale           
                              };

                [obj.NumMCMC,...
                 obj.BurnInRate,...
                 obj.TargetAccept,...
                 obj.NumCovariance,...
                 obj.SaveFileName,...
                 obj.SaveAfter,...
                 obj.Verbose,...
                 obj.Seed,...
                 obj.SigScale,...
                 obj.Scale] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});                
            end
            
            obj.BurnIn = floor(obj.BurnInRate*obj.NumMCMC);
            
            % Set up saved file name
            if isempty(obj.SaveFileName)
                DateVector          = datevec(date);
                [~, MonthString]    = month(date);
                date_time           = [num2str(DateVector(3)),'_',MonthString];
                obj.SaveFileName    = ['MCMC_',model.ModelName,'_',num2str(obj.NumMCMC),'_',date_time,'.mat'];
            end

            % Run MCMC
            obj.Post   = obj.fit(data); 

           % Parameters
           obj.Params = obj.do_mcmc(model);           

        end
        
        % Sample a posterior using MCMC
        function Post = fit(obj,data)
            
            % Extract sampling setting
            model        = obj.Model;
            num_params   = model.NumParams;
            verbose      = obj.Verbose;
            numMCMC      = obj.NumMCMC;
            scale        = obj.Scale;
            V            = obj.SigScale*eye(num_params);
            accept_rate  = obj.TargetAccept;
            N_corr       = obj.NumCovariance;
            saveAfter    = obj.SaveAfter;
            saveFileName = obj.SaveFileName;
            params_init  = obj.ParamsInit;
            
            thetasave    = zeros(numMCMC,num_params);
                         
            % Get initial values of parameters
            if ~isempty(params_init)
                if (length(params_init) ~= num_params)
                    error('MCMC_garch: initial vector with wrong number of parameters.')
                else
                    params = params_init;
                end
            else
                error('MCMC_garch: initial vector of parameters is empty.')
            end
            
            % Make sure params is a row vector
            params = reshape(params,1,num_params);
            
            % For the first iteration
            log_prior = model.logPriors(params);
            lik       = model.logLik(data,params);
            jac       = model.logJac(params);
            post      = log_prior + lik;
            
            tic
            for i=1:numMCMC
                if verbose>0
                    if(mod(i,verbose)==0)
                        disp(['MCMC iter: ',num2str(i),' (',num2str(i/numMCMC*100,'%.0f'),'%).'])
                    end
                end

                % Transform params to normal distribution scale
                params_normal = model.toNormalParams(params);
                
                % Using multivariate normal distribution as proposal distribution
                sample = mvnrnd(params_normal,scale.*V);

                % Convert theta to original distribution
                theta = model.toOriginalParams(sample);

                % Calculate acceptance probability for new proposed sample
                log_prior_star = model.logPriors(theta);
                lik_star       = model.logLik(data,theta);
                jac_star       = model.logJac(theta);
                post_star      = log_prior_star + lik_star;

                A = rand();
                r = exp(post_star-post+jac-jac_star);
                C = min(1,r);   
                if A<=C
                    params = theta;
                    post   = post_star;
                    jac    = jac_star;
                end
                thetasave(i,:) = params;

                % Adaptive scale for proposal distribution
                if i > 50
                    scale = utils_update_sigma(scale,C,accept_rate,i,num_params);
                    if (i > N_corr)
                        V = cov(thetasave(i-N_corr+1:i,:));
                    else
                        V = cov(thetasave(1:i,:));
                    end
                    V = utils_jitChol(V);
                end
                Post.theta(i,:) = params;
                Post.scale(i)   = scale;

                % Store results after each 5000 iteration
                if(saveAfter>0)
                    if mod(i,saveAfter)==0
                        save(saveFileName,'Post')
                    end
                end
            end

            Post.cpu = toc;             
        end
        
        function[mcmc] = do_mcmc(obj,Model)
            mcmc        = getParamsMean(obj);
            mcmc.n      = obj.NumMCMC;
            mcmc.tpar   = mcmc.mu;
            mcmc.par    = Model.itransform(mcmc.tpar);
            mcmc.cov    = cov(mcmc.chain);
            mcmc.Sig2   = diag(mcmc.cov);
        end

        % Function to get parameter means given MCMC samples
        function [MC_est] = getParamsMean(obj,varargin)
            
            post = obj.Post;          
            burnin = obj.BurnIn;

            chain = post.theta(burnin+1:end,:);
            params_mean = mean(chain);
            params_std  = sqrt(mean(chain.^2)-params_mean.^2);
            params_mean = params_mean';
            params_std  = params_std';

            MC_est.mu   = params_mean;
            MC_est.std  = params_std;
            MC_est.par  = obj.Model.itransform(params_mean);
            MC_est.chain = chain;
            
        end

        function[dis] = get_xy_distribution(obj,Num_xp,k,trans)
            NumParams   = obj.Model.NumParams;
            x        = zeros(Num_xp,NumParams);
            y        = zeros(Num_xp,NumParams);

            if ~trans
                c       = obj.Params.chain;
                pmea    = mean(c);
                pstd    = std(c);
                for i = 1:NumParams
                    x(:,i) = linspace(pmea(i)-k*pstd(i),pmea(i)+k*pstd(i),Num_xp);
                    y(:,i) = ksdensity(c(:,i),x(:,i));
                end

            else

                N_chain = size(obj.Params.chain,1);
                c = zeros(N_chain,NumParams);
                for i = 1:N_chain
                    c(i,:) =  obj.Model.itransform(obj.Params.chain(i,:)');
                    if mod(i,floor(N_chain*0.2)) == 0 || i == N_chain
                        fprintf('Transformed mcmc chain: %.1f%%.\n',i/N_chain*100)
                    end
                end

                pmea = mean(c);
                pstd = std(c);

                for i = 1:NumParams
                    x(:,i) = linspace(pmea(i)-k*pstd(i),pmea(i)+k*pstd(i),Num_xp);
                    y(:,i) = ksdensity(c(:,i),x(:,i));
                end
            end
            dis.x = x;
            dis.y = y;
            dis.k = k;
            dis.Num_xp = Num_xp;
            dis.transform = logical(trans);
        end


    end
end

