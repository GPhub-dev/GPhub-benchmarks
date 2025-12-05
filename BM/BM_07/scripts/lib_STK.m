% STK

switch ACTION
    case 'init'
        % INITIALIZE STK
        addpath('~/Documents/MATLAB/stk')
        startup
        
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
        
        DIM = 20;  % 20D for Morris
        model = stk_model(@stk_materncov52_aniso, DIM);  % Matérn 5/2 with ARD
        model.lm = stk_lm_constant;
        model.lognoisevariance = 2 * log (1e-1);  % Noise std = 0.1 (normalized scale)
        param0 = stk_param_init (model, train_x, train_y);
    case 'train'
        % TRAINING
        model.param = stk_param_estim(model, train_x, train_y, param0);
    case 'test'
        % PREDICTION
        test_x_norm = (test_x - mu_x) ./ sigma_x;
        zp = stk_predict(model, train_x, train_y, test_x_norm);
        pred_y = zp.mean * sigma_y + mu_y;
        pred_var = zp.var * sigma_y^2;
end
