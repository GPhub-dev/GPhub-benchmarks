% Benchmark BM_06 (function 5, borehole) - GPstuff library.

switch ACTION
    case 'init'
        % INITIALIZE GPstuff
        gpstuff_root = '~/Documents/MATLAB/gpstuff-develop';
        for sf = {'diag', 'dist', 'gp', 'mc', 'misc', 'optim'}
            addpath(fullfile(gpstuff_root, sf{:}))
        end
        d = size(train_x, 2);
        % normalize inputs and target with the training statistics (data stored in physical units)
        mu_x = mean(train_x, 1); sd_x = std(train_x, 0, 1); sd_x(sd_x == 0) = 1;
        mu_y = mean(train_y); sd_y = std(train_y);
        x_n = (train_x - mu_x) ./ sd_x; y_n = (train_y - mu_y) / sd_y;
        % noise variance estimated, initial value 1e-2
        lik = lik_gaussian('sigma2', 1e-2, 'sigma2_prior', prior_logunif());
        % Matern 5/2 kernel, one length-scale per input.
        % GPstuff optimizes the log-parameters: log-uniform priors cancel the Jacobian, i.e., maximum likelihood
        gpcf = gpcf_matern52('lengthScale', 1.0 * ones(1, d), 'magnSigma2', 1, ...
            'lengthScale_prior', prior_logunif(), 'magnSigma2_prior', prior_logunif());
        % constant trend: constant basis function whose weight has a vague Gaussian prior
        gpmf = gpmf_constant('prior_mean', 0, 'prior_cov', 100);
        gp = gp_set('lik', lik, 'cf', {gpcf}, 'meanf', {gpmf}, 'jitterSigma2', 0);
        opt = optimset('TolFun', 1e-6, 'TolX', 1e-6, 'MaxIter', 1000, 'Display', 'off');
    case 'train'
        % TRAINING (scaled conjugate gradients, the default optimizer of GPstuff)
        gp = gp_optim(gp, x_n, y_n, 'opt', opt);
    case 'test'
        % PREDICTION
        xt_n = (test_x - mu_x) ./ sd_x;
        [Eft, Varft] = gp_pred(gp, x_n, y_n, xt_n);
        pred_y = Eft * sd_y + mu_y;
        pred_var = (Varft + gp.lik.sigma2) * sd_y^2;   % predictive variance of the observations
end
