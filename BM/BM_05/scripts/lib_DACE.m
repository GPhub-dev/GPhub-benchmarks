% DACE

switch ACTION
    case 'init'
        % INITIALIZE DACE
        addpath('~/Documents/MATLAB/dace')
        theta0 = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25];  % ARD: one value for each input dimension (6D)
        lob = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6];  % Lower bounds for 6 dimensions
        upb = [1e6, 1e6, 1e6, 1e6, 1e6, 1e6];  % Upper bounds for 6 dimensions
    case 'train'
        % TRAINING
        [dmodel, perf] = dacefit(train_x, train_y, @regpoly0, @corrgauss, theta0, lob, upb);
    case 'test'
        % PREDICTION
        [pred_y, pred_var] = predictor(test_x, dmodel);
end
