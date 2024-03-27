function[lbl] = mk_lbl(obj)

GarchType = obj.GarchType;
p = obj.P;
o = obj.O;
q = obj.Q;

switch GarchType
    case 'garch'       
       a= arrayfun(@(i) ['alpha_' num2str(i)],1:p,'uni',0);
       g= arrayfun(@(i) ['gamma_' num2str(i)],1:o,'uni',0);
       b= arrayfun(@(i) ['beta_' num2str(i)],1:q,'uni',0);
       lbl_par = {{'w'},a,g,b};
       lbl_par = cat(2,lbl_par{:});

    case 'egarch'
       a= arrayfun(@(i) ['alpha_' num2str(i)],1:p,'uni',0);
       g= arrayfun(@(i) ['gamma_' num2str(i)],1:o,'uni',0);
       b= arrayfun(@(i) ['beta_' num2str(i)],1:q,'uni',0);
       lbl_par = {{'w'},a,g,b};
       lbl_par = cat(2,lbl_par{:});

    case 'figarch'
       phi= arrayfun(@(i) ['phi_' num2str(i)],1:p,'uni',0);
       d= arrayfun(@(i) ['d_' num2str(i)],1:1,'uni',0);
       b= arrayfun(@(i) ['beta_' num2str(i)],1:q,'uni',0);
       lbl_par = {{'w'},phi,d,b};
       lbl_par = cat(2,lbl_par{:});
end




lbl_perf = {'lb','nll','mse','mse_rv','qlik','qlik_rv'};

lbl_perf_tr = cellfun(@(c) ['tr_' c],lbl_perf,'uni',0);
lbl_perf_te = cellfun(@(c) ['te_' c],lbl_perf,'uni',0);


lbl = [lbl_par,lbl_perf_tr,lbl_perf_te];

end