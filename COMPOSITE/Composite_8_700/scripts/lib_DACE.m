% Composite plate - DACE (harmonized specification, see _scripts/README.md)

switch ACTION
    case 'init'
        % INITIALIZE DACE
        addpath('~/Documents/MATLAB/dace')
        m = size(train_x, 2);
        l0 = 0.25 * sqrt(m);
        % DACE: R = exp(-sum theta_k h_k^2) on inputs standardized to unit variance -> theta = var(x) / (2 l^2)
        var_x = 1 / 12;
        theta0 = (var_x / (2 * l0^2)) * ones(1, m);   % one value per input (anisotropic)
        lob = (var_x / (2 * 1e2^2)) * ones(1, m);     % l = 1e2
        upb = (var_x / (2 * 1e-2^2)) * ones(1, m);    % l = 1e-2
        % DACE exposes no nugget: it adds (10 + n) * eps to the diagonal
    case 'train'
        % TRAINING
        [dmodel, perf] = dacefit(train_x, train_y, @regpoly0, @corrgauss, theta0, lob, upb);
    case 'test'
        % PREDICTION
        [pred_y, pred_var] = predictor(test_x, dmodel);
end
