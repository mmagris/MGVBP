function[nll,ht] = garch_nll_ht(data,par,obj)

switch obj.GarchType
    case 'garch'
        [nll,ht]   = fun_tarch_nll(data,par,obj.P,obj.O,obj.Q,1);
    case 'egarch'
        [nll,ht]   = fun_egarch_nll(data,par,obj.P,obj.O,obj.Q,1);
    case 'figarch'
        [nll,ht]   = fun_figarch_nll(data,par,obj.P,obj.Q,0.5,1);
    case 'rgarch'
        [nll,ht]   = fun_rgarch_nll(data,par,obj.P,obj.Q,1);
end

end