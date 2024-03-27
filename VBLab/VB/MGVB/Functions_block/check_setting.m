function[setting, useHfunc,UseInvChol] = check_setting(setting)

if ~isfield(setting,'UseInvChol')
    setting.UseInvChol = 0;
else
    if ~(setting.UseInvChol == 0 || setting.UseInvChol == 1)
        error('Invalid setting.UseInvChol field in setting.')
    end
end

if ~isfield(setting,'doCV')
    setting.doCV = 1;
else
    if ~(setting.doCV == -1 || setting.doCV == 0)
        error('Invalid setting.doCV field in setting.')
    end
end

if  ~isfield(setting,'useHfunc')
    error('Missing useHfunc field in setting.')
end

useHfunc    = setting.useHfunc;
UseInvChol  = setting.UseInvChol;

if size(setting.Prior.Mu,2)>size(setting.Prior.Mu,1)
    setting.Prior.Mu = setting.Prior.Mu';
end
setting.mu0 = setting.Prior.Mu;

if numel(setting.mu0)>1 && numel(setting.mu0)~= setting.d_theta
    error('Size of parameters in Prior.Mu does not match the number of parameters.')
end

end