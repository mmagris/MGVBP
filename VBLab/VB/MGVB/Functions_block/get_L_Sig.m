function[L,iS] = get_L_Sig(S,setting)

    if setting.UseInvChol == 0
        iS = inv(S);
        L = chol(S,'lower');
    elseif setting.UseInvChol == 1
        L = chol(S,'lower');
        iL = inv(L);
        iS = iL'*iL;
    end


end
