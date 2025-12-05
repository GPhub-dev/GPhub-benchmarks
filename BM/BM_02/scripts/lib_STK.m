% STK

switch ACTION
    case 'init'
        % INITIALIZE STK
        addpath('~/Documents/MATLAB/stk')
        startup
        DIM = 2;
        model = stk_model(@stk_materncov52_aniso, DIM);  % Matérn 5/2 with ARD
        model.lm = stk_lm_constant;
        model.lognoisevariance = 2 * log (1e-4);  % Noise std = 1e-4 (small)
        param0 = stk_param_init (model, train_x, train_y);
    case 'train'
        % TRAINING
        model.param = stk_param_estim(model, train_x, train_y, param0);
    case 'test'
        % PREDICTION
        zp = stk_predict(model, train_x, train_y, test_x);
        pred_y = zp.mean;
        pred_var = zp.var;
end
