
function[Data] = mk_train_test_data(data,split)

n_data      = size(data,1);

X = data(:,2:end);
Y = data(:,1);
O = ones(n_data,1);

data = [Y,O,X];


n_train     = floor(n_data*split);
n_test      = n_data-n_train;

Data.all        = data;
Data.train      = data(1:n_train,:);
Data.test       = data(n_train+1:end,:);
Data.n.all      = n_data;
Data.n.train    = n_train;
Data.n.test     = n_test;
Data.split      = split;
end