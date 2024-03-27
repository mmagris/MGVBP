function[ll] = glm_ll(data,theta)
X = data(:,1:end-1);
y = data(:,end);
YProb = 1./(1+exp(-X*theta));
ll = sum(y.*log(YProb)+(1-y).*log(1-YProb));
end