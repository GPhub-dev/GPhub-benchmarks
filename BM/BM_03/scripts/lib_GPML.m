% GPML

switch ACTION
    case 'init'
        % INITIALIZE GPML
        addpath('~/Documents/MATLAB/gpml-matlab-v4.2-2018-06-11')
        startup
        meanfunc = @meanConst;           % Constant mean function
        covfunc = {@covMaternard, 5};     % Matérn 5/2 with ARD (one length scale per dim)
        likfunc = @likGauss;             % Gaussian likelihood
        hyp = struct();
        hyp.mean = 0;                    % Initial value for constant mean function
        hyp.cov = [0.25; 0.25; 1];       % ARD: 2 length scales + signal variance
        hyp.lik = log(sqrt(1e-8));       % Initial noise variance (log of std dev)
    case 'train'
        % TRAINING
        hyp_opt = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, train_x, train_y);
    case 'test'
        % PREDICTION
        [pred_y, pred_var] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, train_x, train_y, test_x);
end
