function[gradLB_mu,gradLB_iSig] = nat_gradients(setting,Y12,mu,Sig,Sig_inv)

useHfunc    = setting.useHfunc;
Sig0_type   = setting.Sig0_type;
mu0         = setting.mu0;
Sig_inv_0   = setting.Sig_inv_0;

B = setting.Block;
Nb = B.n;

gradLB_mu = zeros(setting.d_theta,1);
gradLB_iSig = cell(Nb,1);

C_S     = cell(Nb,1);
C_mu    = cell(Nb,1);

iF = setting.iFullPrior;
cond_iF = all(all(~isnan(iF)));

aux = (mu-mu0);
if  cond_iF && useHfunc == 0
    tmp = iF*aux;
end



for b = 1:Nb
    indx_1 = B.indx{b,1};

    % Same as MGVB
    if cond_iF
        if useHfunc
            C_mu{b} = 0;
        else
            C_mu{b} = -Sig{b}*tmp(indx_1);
        end

    else

        if useHfunc
            C_mu{b} = 0;
        else
            % Same as MGVB
            if Sig0_type(b) == 1
                C_mu{b}    = -Sig{b}*Sig_inv_0{b}*aux(indx_1);
            elseif Sig0_type(b) == 2
                C_mu{b}    = -Sig{b}*(Sig_inv_0{b}.*aux(indx_1));
            else
                C_mu{b}    = -Sig{b}*(Sig_inv_0{b}.*aux(indx_1));
            end
        end
    end
    
    % Same as MGVB
    gradLB_mu(indx_1,1)  = C_mu{b} + Y12(indx_1);

end


for b = 1:Nb

    d_theta_b = B.blks(b);
    indx_2 = B.indx{b,3};


    if useHfunc
        C_S{b} = 0;
        C_mu{b} = 0;
    else
        % This is different than MGVB (MGVB constants multiplied by -0.5)
        if Sig0_type(b) == 1 
            C_S{b}     = Sig_inv_0{b}-Sig_inv{b};
        elseif Sig0_type(b) == 2 
            C_S{b}     = diag(Sig_inv_0{b}) -Sig_inv{b};
        else 
            C_S{b}     = eye(d_theta_b)*Sig_inv_0{b} -Sig_inv{b};
        end

    end

    % This is different than MGVB
    gradLB_iSig{b,1}    = C_S{b}  + reshape(Y12(indx_2),d_theta_b,d_theta_b);
    

end



end



