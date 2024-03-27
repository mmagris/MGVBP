function[opt] = set_opt(GarchType,theta_init)

switch GarchType
    case 'garch'
        opt.lr              = 0.01;
        opt.maxiter         = 500;
        opt.maxPatience     = 5000;
        opt.stepAdaptive    = 1000;
        opt.maxGrad         = 1000;
        opt.nSample         = 150;
        opt.clipInit        = 100;
        opt.MeanInit        = theta_init;
        opt.SigInitScale    = 0.05;

    case 'egarch'
        opt.lr              = 0.01;
        opt.maxiter         = 3000;
        opt.maxPatience     = 5000;
        opt.stepAdaptive    = 2500;
        opt.maxGrad         = 1000;
        opt.nSample         = 150;
        opt.clipInit        = 1000;
        opt.MeanInit        = theta_init;
        opt.SigInitScale    = 0.05;

    case 'figarch'
        opt.lr              = 0.01;
        opt.maxiter         = 1200;
        opt.maxPatience     = 5000;
        opt.stepAdaptive    = 2500;
        opt.maxGrad         = 1000;
        opt.nSample         = 150;
        opt.clipInit        = 1000;
        opt.MeanInit        = theta_init;
        opt.SigInitScale    = 0.05;
        
    otherwise
        error('Invalid GarchType.')
end

opt.TrainTest       = 1;
opt.SaveParams      = 1;
opt.useHfunc        = 1;
opt.Verbose         = 2;
end
