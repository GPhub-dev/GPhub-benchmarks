% Composite plate - UQLab (harmonized specification, see _scripts/README.md)

switch ACTION
    case 'init'
        % INITIALIZE UQLab
        addpath(fullfile('~/Documents/MATLAB/UQLab2.0', 'core'))
        uqlab;
        m = size(train_x, 2);
        l0 = 0.25 * sqrt(m);
        MetaOpts.Type = 'Metamodel';
        MetaOpts.MetaType = 'Kriging';
        MetaOpts.Trend.Type = 'ordinary';          % constant trend, estimated
        MetaOpts.Corr.Family = 'Gaussian';         % Gaussian kernel
        MetaOpts.Corr.Type = 'ellipsoidal';
        MetaOpts.Corr.Isotropic = false;           % one length-scale per input
        MetaOpts.Corr.Nugget = 1e-8;               % fixed nugget
        MetaOpts.Scaling = false;                  % inputs are already in [0, 1]: length-scales in these units
        MetaOpts.ExpDesign.X = train_x;
        MetaOpts.ExpDesign.Y = train_y;
        MetaOpts.EstimMethod = 'ML';
        MetaOpts.Optim.Method = 'BFGS';
        MetaOpts.Optim.InitialValue = l0 * ones(m, 1);
        MetaOpts.Optim.Bounds = [1e-2 * ones(1, m); 1e2 * ones(1, m)];
        MetaOpts.Optim.MaxIter = 1000;
    case 'train'
        % TRAINING
        myKriging = uq_createModel(MetaOpts);
    case 'test'
        % PREDICTION
        [pred_y,pred_var] = uq_evalModel(myKriging,test_x);
end
