function[setting, SigFactor] = Factor_settings(setting)

if ~isfield(setting,'SigFactor')
    setting.SigFactor = 1;
else
    if setting.SigFactor ==0
        error('Invalid setting.SigFactor.')
    end
end
SigFactor = setting.SigFactor;
end