function[ssave] = create_iter_struct(save_params,max_iter,setting)

ssave = struct();
d_theta = setting.d_theta;

n_sig = sum(setting.Block.blks.^2);

if(save_params)
    ssave.mu    = zeros(max_iter,d_theta);
    ssave.ll    = zeros(max_iter,1);
    ssave.logq  = zeros(max_iter,1);
    ssave.SigInv = zeros(max_iter,n_sig);


end
end