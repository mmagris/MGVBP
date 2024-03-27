function[tpar] = garch_transform(par,p,o,q,type)

switch type
    case 'garch'
        tpar = tarch_transform(par,p,o,q,1);
    case 'egarch'
        tpar = egarch_transform(par,p,o,q,1);
    case 'figarch'
        tpar = figarch_transform(par,p,q,1);
    case 'rgarch'
        %Only for p = 1, q = 1;
        tpar = rgarch_itransform(par,p,q);
end


end
