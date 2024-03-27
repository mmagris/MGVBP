function[nLL,ht] = fun_egarch_nll(data,par,p,o,q,isTrans)


m  = max([p o q]);
data_augmented=[zeros(m,1);data];
T  = size(data_augmented,1);

back_cast = 0.7;
error_type = 1; %NORMAL, code works only for normal!

% back_cast_length = max(floor(length(data)^(1/2)),1);
% back_cast_weights = .05*(.9.^(0:back_cast_length ));
% back_cast_weights = back_cast_weights/sum(back_cast_weights);
% back_cast = back_cast_weights*(data(1:back_cast_length+1).^2);
% if back_cast==0
%     back_cast=log(cov(data));
% else
%     back_cast=log(back_cast);
% end
% back_cast
[nLL, ~, ht] = egarch_likelihood(par, data_augmented, p, o, q, error_type, back_cast, T, isTrans);

end

