function[par] = garch_itransform(tpar,p,o,q,type)

switch type
    case 'garch'
        par = tarch_itransform(tpar,p,o,q,1);
    case 'egarch'
        par = egarch_itransform(tpar,p,o,q,1);
    case 'figarch'
        par = figarch_itransform(tpar,p,q,1);
    case 'rgarch'
        %Only for p = 1, q = 1;
        par = rgarch_itransform(tpar,p,q);
end


end