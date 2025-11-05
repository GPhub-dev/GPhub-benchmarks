% GPML

switch ACTION
    case 'init'
        % INITIALIZE GPML
        addpath('~/Documents/MATLAB/gpml-matlab-v4.2-2018-06-11')
        startup
        meanfunc = []            % Constant mean function
        covfunc = @covSEiso;             % Squared Exponential (RBF) kernel
        likfunc = @likGauss;             % Gaussian likelihood
        hyp = struct();
        % hyp.mean = [0; 0];                   % No hyperparameters for the zero mean function
        hyp.mean = [];                   % No hyperparameters for the zero mean function
        hyp.cov = [0.25; 1];                % Initial guess for length scale and signal variance
        hyp.lik = 1e-10;%log(0.01);             % Initial noise variance
    case 'train'
        % TRAINING
        hyp_opt = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, train_x, train_y);
    case 'test'
        % PREDICTION
        [pred_y, pred_var] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, train_x, train_y, test_x);
end
