% DACE

switch ACTION
    case 'init'
        % INITIALIZE DACE
        addpath('~/Documents/MATLAB/dace')
        
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
        
        % ARD: one value for each input dimension (20D)
        % Since data is normalized, theta (activity param ~ 1/length_scale^2) 
        % should be around 1.0. For inert variables, theta should be close to 0.
        theta0 = ones(1, 20) * 1.0; 
        lob = ones(1, 20) * 1e-6;  % Lower bounds (very inactive)
        upb = ones(1, 20) * 1e2;   % Upper bounds (very active)
    case 'train'
        % TRAINING
        [dmodel, perf] = dacefit(train_x, train_y, @regpoly0, @corrgauss, theta0, lob, upb);
    case 'test'
        % PREDICTION
        test_x_norm = (test_x - mu_x) ./ sigma_x;
        [pred_y_norm, pred_var_norm] = predictor(test_x_norm, dmodel);
        
        pred_y = pred_y_norm * sigma_y + mu_y;
        pred_var = pred_var_norm * sigma_y^2;
end
