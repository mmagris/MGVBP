function[mat,LL_mfe] = fit_garch_ml(data,p,o,q,type)

switch type

    case 'garch'
        if o~=0
            warning('Estimating gjr.')
            EstMdl  = estimate(gjr('GARCHLags',1:q,'ARCHLags',1:p,'LeverageLags',1:o),data);
            par_mat = [EstMdl.Constant,EstMdl.ARCH{:},EstMdl.Leverage{:},EstMdl.GARCH{:}]';
        else
            EstMdl  = estimate(garch('GARCHLags',1:q,'ARCHLags',1:p),data);
            par_mat = [EstMdl.Constant,EstMdl.ARCH{:},EstMdl.GARCH{:}]';
        end

        tpar_mat            = garch_transform(par_mat,p,o,q,type);
        [par_mfe,LL_mfe]    = tarch_mfe(data,p,o,q);
        tpar_mfe            = garch_transform(par_mfe,p,o,q,type);

    case 'egarch'
        EstMdl              = estimate(egarch('GARCHLags',1:q,'ARCHLags',1:p,'LeverageLags',1:o),data);
        par_mat             = [EstMdl.Constant,EstMdl.ARCH{:},EstMdl.Leverage{:},EstMdl.GARCH{:}]';
        tpar_mat            = garch_transform(par_mat,p,o,q,type);

        [par_mfe,LL_mfe]    = egarch_mfe(data,p,o,q);
        tpar_mfe            = garch_transform(par_mat,p,o,q,type);

    case 'figarch'
        [par_mfe,LL_mfe]    = figarch_mfe(data,p,q);
        tpar_mfe            = garch_transform(par_mfe,p,o,q,type);

        par_mat             = par_mfe*nan;
        tpar_mat            = tpar_mfe*nan;

    case 'rgarch'
        load('GARCH\SPYData.mat')
        options             = optimset('Display','off','MaxFunEvals',15000,'MaxIter',15000,'TolFun',1e-9,'TolX',1e-9);
        theta_init          = zeros(9,1)+0.05;
        f                   = @(x) rgarch_likelihood(x,1,1,data(:,1),data(:,2),0,1);
        [tpar_mfe,fval]     = fminsearch(f,theta_init,options);
        par_mfe             = garch_itransform(tpar_mfe,p,o,q,type);
        LL_mfe              = -fval;

        par_mat             = Es(1:9);
        tpar_mat            = garch_transform(par_mat,p,o,q,type);

    otherwise
        error('Wrong GARCH model. Use garch, egarch, figarch, rgarch.')
end

mat = [par_mfe,par_mat,tpar_mfe,tpar_mat];

array2table(mat,'VariableNames',{'par_mfe','par_mat','tpar_mfe','tpar_mat'})

end