% UQLab

switch ACTION
    case 'init'
        % INITIALIZE UQLab
        uqlab;
        MetaOpts.Type = 'Metamodel';
        MetaOpts.MetaType = 'Kriging';
        MetaOpts.Trend.Type = 'ordinary';
        MetaOpts.Corr.Family = 'Gaussian';
        MetaOpts.Corr.Nugget = 1e-10;
        MetaOpts.ExpDesign.X = train_x;
        MetaOpts.ExpDesign.Y = train_y;
        MetaOpts.EstimMethod = 'ML';
        MetaOpts.Optim.Method = 'BFGS';
        MetaOpts.Optim.InitialValue = 0.25;
        MetaOpts.Optim.Bounds = [1e-6; 1e6];
        MetaOpts.Optim.MaxIter = 1000;
    case 'train'
        % TRAINING
        myKriging = uq_createModel(MetaOpts);
    case 'test'
        % PREDICTION
        [pred_y,pred_var] = uq_evalModel(myKriging,test_x);
end
