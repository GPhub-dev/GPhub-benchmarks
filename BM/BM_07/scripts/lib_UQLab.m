% UQLab

switch ACTION
    case 'init'
        % INITIALIZE UQLab
        uqlab;
        
        % Normalize inputs
        mu_x = mean(train_x);
        sigma_x = std(train_x);
        sigma_x(sigma_x==0) = 1;
        train_x = (train_x - mu_x) ./ sigma_x;
        
        % Normalize target
        mu_y = mean(train_y);
        sigma_y = std(train_y);
        if sigma_y==0, sigma_y=1; end
        train_y = (train_y - mu_y) ./ sigma_y;
        
        MetaOpts.Type = 'Metamodel';
        MetaOpts.MetaType = 'Kriging';
        MetaOpts.Trend.Type = 'ordinary';
        MetaOpts.Corr.Family = 'Matern-5_2';  % Matérn 5/2 kernel
        MetaOpts.Corr.Nugget = 1e-5;  % small nugget
        MetaOpts.ExpDesign.X = train_x;
        MetaOpts.ExpDesign.Y = train_y;
        MetaOpts.EstimMethod = 'ML';
        MetaOpts.Optim.Method = 'BFGS';
        MetaOpts.Optim.InitialValue = 1.0;
        MetaOpts.Optim.Bounds = [1e-2; 1e3]; % Normalized length scales
        MetaOpts.Optim.MaxIter = 1000;
    case 'train'
        % TRAINING
        myKriging = uq_createModel(MetaOpts);
    case 'test'
        % PREDICTION
        test_x_norm = (test_x - mu_x) ./ sigma_x;
        [pred_y_norm, pred_var_norm] = uq_evalModel(myKriging, test_x_norm);
        pred_y = pred_y_norm * sigma_y + mu_y;
        pred_var = pred_var_norm * sigma_y^2;
end
