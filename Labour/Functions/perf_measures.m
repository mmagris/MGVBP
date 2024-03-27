
function[metrics] = perf_measures(data,theta)

X = data(:,1:end-1);
y = data(:,end);

ypred = predict_glm(X,theta);

metrics = APRF(y,ypred);

end

function[metrics] = APRF(Y_true,Y)

C = confusionmat(Y_true,Y);
pre = mean(diag(C)./sum(C,1)');
rec = mean(diag(C)./sum(C,2));
acc = sum(diag(C))./sum(sum(C));
f1 = 2*rec*pre/(rec+pre);

metrics = [acc,pre,rec,f1];

end

function[Ylbl,YProb] = predict_glm(X,par)

    aux = X*par;
    YProb = 1./(1+exp(-aux));
    Ylbl = YProb;
    Ylbl(Ylbl>=0.5) = 1;
    Ylbl(Ylbl<0.5) = 0;

end