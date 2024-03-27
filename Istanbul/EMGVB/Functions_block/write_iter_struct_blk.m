function[ssave] = write_iter_struct_blk(ssave,save_params,iter,mu,llh_s,log_q_lambda_s,Sig_inv)

if(save_params)
    ssave.mu(iter,:)     = mu;
    ssave.ll(iter,:)     = mean(llh_s);
    ssave.logq(iter,:)   = mean(log_q_lambda_s);

    nb = numel(Sig_inv);
    tmp_2 = cell(nb);

    for b = 1:nb
        tmp      = Sig_inv{b};
        tmp_2{b} = tmp(:);
    end
    ssave.SigInv(iter,:) = vertcat(tmp_2{:})';
end
end