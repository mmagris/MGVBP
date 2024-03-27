classdef mcmc_garch_mdl
    %LOGISTICREGRESSION Summary of this class goes here
    %   Detailed explanation goes here
    % Attributes
    properties
        ModelName      % Model name
        NumParams      % Number of parameters
        PriorInput     % Prior specified by users
        Prior          % Prior object
        PriorVal       % Parameters of priors
        Intercept      % Option to add intercept or not (only for testing)
        AutoDiff       % Option to use autodiff (only for testing)
        CutOff         % Cutoff probability for classification making
        Post           % Struct to store training results (maybe not used)
        P
        O
        Q
        GarchType
        ParamsInit
        InitMethod        
    end

    methods
        % Constructors
        function obj = mcmc_garch_mdl(varargin)
            %LOGISTICREGRESSION Construct an instance of this class
            %   Detailed explanation goes here

            obj.PriorInput  = {'Normal',[0,1]};
            obj.Intercept   = false;
            obj.AutoDiff    = true;
            obj.CutOff      = 0.5;

            % Get additional arguments (some arguments are only for testing)
            if nargin > 1
                %Parse additional options
                paramNames = {  'AutoDiff'      'Intercept'     'Prior'         'CutOff',...
                    'P'             'O'             'Q'             'GarchType',...
                    'ParamsInit'};

                paramDflts = {  obj.AutoDiff    obj.Intercept   obj.PriorInput  obj.CutOff,...
                    obj.P           obj.O           obj.Q           obj.GarchType,...
                    obj.ParamsInit};

                [obj.AutoDiff,...
                    obj.Intercept,...
                    obj.PriorInput,...
                    obj.CutOff,...
                    obj.P,...
                    obj.O,...
                    obj.Q,...
                    obj.GarchType,...
                    obj.ParamsInit,...
                    ] = internal.stats.parseArgs(paramNames, paramDflts, varargin{:});
            end


            switch obj.GarchType
                case 'garch'
                    obj.NumParams   = obj.P + obj.O + obj.Q + 1;
                    
                case 'egarch'
                    obj.NumParams   = obj.P + obj.O + obj.Q + 1;

                case 'figarch'
                    if obj.O~=0
                        warning('Estimating figarch: ignoring o.')
                    end
                    obj.NumParams   = obj.P + obj.Q + 2;
                    obj.O = nan;

                case 'rgarch'
                    if obj.P~=1
                        warning('Estimating rgarch: setting p to 1.')
                        obj.P = 1;
                    end
                    if obj.Q~=1
                        warning('Estimating rgarch: setting q to 1.')
                        obj.Q = 1;
                    end
                    if obj.O~=0
                        warning('Estimating rgarch: ignoring o.')
                    end

                    obj.NumParams   = obj.P + obj.Q + 7;
                    obj.O = nan;

                otherwise
                    error('Invalid GarchType: use garch, egarch, figarch or rgarch.')
            end

            obj.ModelName   = [obj.GarchType '_' num2str(obj.P) '_' num2str(obj.O) '_' num2str(obj.Q)];

            if isempty(obj.ParamsInit)
                obj.ParamsInit  = normrnd(0,2,1,obj.NumParams);
                obj.InitMethod  = 'random';

            elseif obj.ParamsInit == 0

                obj.ParamsInit = zeros(1,obj.NumParams);
                obj.InitMethod  = 'zeros';

            else
                obj.InitMethod  = 'fixed';
                if obj.NumParams ~= numel(obj.ParamsInit)
                    error(['ParamsInit mu must be of dimension (1x', num2str(obj.NumParams,'%i') ')!'])
                end
            end

            % Set prior object using built-in distribution classes
            eval(['obj.Prior=',obj.PriorInput{1}]);
            obj.PriorVal = obj.PriorInput{2};

        end

        % Inverse transform of Volatility parameters
        function[tpar] = itransform(obj,par)
            tpar = garch_itransform(par,obj.P,obj.O,obj.Q,obj.GarchType);
        end




        %% Log likelihood
        % Input:
        %   - data: 2D array. The last column is the responses
        %   - params: Dx1 vector of parameters
        % Output:
        %   - llh: Log likelihood of the model
        function llh = logLik(obj,data,params)

            % Make sure params is a columns
            params = reshape(obj.toOriginalParams(params),obj.NumParams,1);

            switch obj.GarchType
                case 'garch'
                    llh = -fun_tarch_nll(data(:,1),params,obj.P,obj.O,obj.Q,1);
                case 'egarch'
                    llh = -fun_egarch_nll(data(:,1),params,obj.P,obj.O,obj.Q,1);
                case 'figarch'
                    llh = -fun_figarch_nll(data(:,1),params,obj.P,obj.Q,[],1);
                case 'rgarch'
                    llh = -fun_rgarch_nll(data(:,1),params,obj.P,obj.Q,1);
            end

        end

        %% Compute gradient of Log likelihood using AutoDiff
        % Input:
        %   - data: 2D array. The last column is the responses
        %   - params: 1xD vector of parameters
        % Output:
        %   - llh_grad: Log likelihood of the model
        function [llh_grad,llh] = logLikGradAutoDiff(obj,data,params)

            llh = obj.logLik(data,params);

            llh_grad = dlgradient(llh,params);
        end

        %% Compute log prior of parameters
        % Input:
        %   - params: the Dx1 vector of parameters
        % Output:
        %   - llh: Log prior of model parameters
        function log_prior = logPriors(obj,params)

            params = reshape(obj.toOriginalParams(params),obj.NumParams,1);

            % Compute log prior
            log_prior = obj.Prior.logPdfFnc(params,obj.PriorVal);

        end

        %% Compute gradient of log prior of parameters
        % Input:
        %   - params: 1xD vector of parameters
        % Output:
        %   - log_prior_grad: Gradient of log prior of model parameters
        function [log_prior_grad,log_prior] = logPriorsGrad(obj,params)

            % Compute log prior
            log_prior = obj.Prior.logPdfFnc(params,obj.PriorVal);

            % Compute gradient of log prior
            log_prior_grad = obj.Prior.GradlogPdfFnc(params,obj.PriorVal);
        end

        %% Log of Jacobian of all paramters
        % Input:
        %   - params: the ROW vector of parameters
        % Output:
        %   - llh: Log prior of model parameters
        function logjac = logJac(obj,params)
            logjac = 0;
        end

        %% Log of Jacobian of all paramters
        % Input:
        %   - params: the ROW vector of parameters
        % Output:
        %   - llh: Log prior of model parameters
        function [logJac_grad,logJac] = logJacGrad(obj,params)
            logJac_grad = 0;
            logJac      = 0;
        end

        %% Function to compute h_theta = log lik + log prior
        % Input:
        %   - data: 2D array. The last column is the responses
        %   - theta: Dx1 vector of parameters
        % Output:
        %   - h_func: Log likelihood + log prior
        function h_func = hFunction(obj,data,theta)
            % Transform parameters from normal to original distribution
            params = obj.toOriginalParams(theta);

            % Compute h(theta)
            log_lik = obj.logLik(data,params);
            log_prior = obj.logPriors(params);
            log_jac = obj.logJac(params);
            h_func = log_lik + log_prior + log_jac;
        end

        %% Function to compute gradient of h_theta = grad log lik + grad log prior
        % Input:
        %   - data: 2D array. The last column is the responses
        %   - theta: Dx1 vector of parameters
        % Output:
        %   - h_func_grad: gradient (Log likelihood + log prior)
        %   - h_func: Log likelihood + log prior
        function [h_func_grad, h_func] = hFunctionGrad(obj,data,theta)

            % Transform parameters from normal to original distribution
            params = obj.toOriginalParams(theta);

            % Compute h(theta)
            [llh_grad,llh] = obj.logLikGrad(data,params);
            [log_prior_grad,log_prior] = obj.logPriorsGrad(params);
            [logJac_grad,logJac] = obj.logJacGrad(params);
            h_func = llh + log_prior + logJac;
            h_func_grad = llh_grad + log_prior_grad + logJac_grad;
        end

        %% Transform parameters to from normal to original distribution
        function paramsOriginal = toOriginalParams(obj,params)
            paramsOriginal = obj.Prior.toOriginalParams(params);
        end

        % Transform parameters to from normal to original distribution
        function paramsNormal = toNormalParams(obj,params)
            paramsNormal = obj.Prior.toNormalParams(params);
        end

end
end

