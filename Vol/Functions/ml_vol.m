
function[ml] = ml_vol(Data,obj)

[tab_ml,~]  = fit_garch_ml(Data.train(:,1),obj.P,obj.O,obj.Q,obj.GarchType);
ml.tab_ml   = tab_ml(:,{'par_mfe','tpar_mfe'});
ml.par      = ml.tab_ml.par_mfe;
ml.tpar     = ml.tab_ml.tpar_mfe;

ml.perf.train = garch_perf(obj,Data.train,ml.tpar);
ml.perf.test  = garch_perf(obj,Data.test,ml.tpar);

end

