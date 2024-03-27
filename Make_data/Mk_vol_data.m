clearvars -except wd
clc


wd = 'C:\Users\Martin\Desktop\EMGVB_REV\Make_data';
cd(wd)
[C,Lab] = Import_oxfordRM('oxfordmanrealizedvolatilityindices.csv');



date_from   = datetime('2014-Jan-01');
date_to     = datetime('2023-Dec-31');

Stock_list = ["SPX"];
RM = struct();

for i = 1:numel(Stock_list)

    Stock = Stock_list(i);

    s_indx = ismember(C{2},Stock);
    t_indx = C{1} >= date_from & C{1} <= date_to;
    Z = cellfun(@(c) c(s_indx & t_indx),C,'uni',0);

    tab = table(Z{1:end});
    tab.Properties.VariableNames = Lab.raw;
    tab.r = [nan;price2ret(tab.close_price,Method="continuous")]; %or use "periodic" for net ret
    tab = tab(:,{'Time','r','rv5_ss','rk_parzen'});

    TT = table2timetable(tab);
    TT(1,:) = [];
    TT.Time.Format = 'yyyy-MMM-dd';

    HAR.(Stock) = mk_har_tab(sqrt(TT.rv5_ss*252)*100,TT.Time); %std deviations *100

    rd     = TT.r-mean(TT.r,'omitnan'); 
    crv5ss = rv2sig(rd,TT.rv5_ss);
    crkp   = rv2sig(rd,TT.rk_parzen);

    GARCH.(Stock) = timetable(TT.Time, rd*100, crkp.^0.5*100, crv5ss.^0.5*100, 'VariableNames',{'r','rv','rk'}); %these are std not variances

end


save('HARdata.mat','HAR')
save('GARCHdata.mat','GARCH')



%%

function[sig2,c] = rv2sig(r_demean,rv)
    % Hansen, P.R. and Lunde, A. (2005), A forecast comparison of volatility models: does anything beat a GARCH(1,1)?.
    % J. Appl. Econ., 20: 873-889. https://doi.org/10.1002/jae.800
    % Equation (1)
    c = mean(r_demean.^2,'omitnan')/mean(rv,'omitnan');
    sig2 = rv*c;

end


function[har] = mk_har_tab(x,Times)
% x = RV.(Stock).RV5;
N = size(x,1);
mat = nan(N,4);
mat(:,1) = x;

for i = 2:N
    mat(i,2) = mat(i-1,1);
end

for i = 6:N
    mat(i,3) = mean(mat(i-5:i-1,1));
end

for i = 23:N
    mat(i,4) = mean(mat(i-22:i-1,1));
end

har = array2timetable(mat,'RowTimes',Times,'VariableNames',{'t','y','w','m'});
har = har(23:end,:);
end




function[C,Lab] = Import_oxfordRM(filename)

fileID = fopen(filename);
h = fgets(fileID);
CC = textscan(fileID,'%10{yyyy-MM-dd}D %*s %*1s %s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f',...
    'Delimiter',',','HeaderLines',1);
fclose(fileID);

h = strsplit(strtrim(string(h)),',');
h(1) = "Time";

Lab.raw = ["Time","Symbol","rsv","rk_parzen","rsv_ss","bv_ss","medrv","open_price","bv","rv5_ss","rv5","open_to_close","rv10_ss","rv10","nobs","close_price","rk_th2","rk_twoscale","open_time","close_time"]';

[~,idx] = ismember(Lab.raw,h);
Lab.order = idx;
C = CC(Lab.order);

end