function[nll,ht] = fun_figarch_nll(data,par,p,q,backCast,isTrans)

if isempty(backCast)
    backCastLength = max(floor(length(data)^(1/2)),1);
    backCastWeights = .05*(.9.^(0:backCastLength ));
    backCastWeights = backCastWeights/sum(backCastWeights);
    backCast = backCastWeights*((data(1:backCastLength+1)).^2);
    if backCast==0
        backCast=cov(data);
    end
end

truncLag    = 1000;
errorType   = 1;

epsilon2Augmented = [zeros(truncLag,1);data.^2];
epsilon2Augmented(1:truncLag) = backCast;

[nll,~,ht] = figarch_likelihood(par,p,q,data,epsilon2Augmented,truncLag,errorType,isTrans);

end

