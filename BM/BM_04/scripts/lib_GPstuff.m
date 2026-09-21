% Benchmark BM_04 (function 3, Dette & Pepelyshev) - GPstuff library.

switch ACTION
    case 'init'
        % INITIALIZE GPstuff
        gpstuff_root = '~/Documents/MATLAB/gpstuff-develop';
        for sf = {'diag', 'dist', 'gp', 'mc', 'misc', 'optim'}
            addpath(fullfile(gpstuff_root, sf{:}))
        end
        d = size(train_x, 2);
        mu_x = zeros(1, d); sd_x = ones(1, d); mu_y = 0; sd_y = 1;   % data already standardized
        x_n = (train_x - mu_x) ./ sd_x; y_n = (train_y - mu_y) / sd_y;
        % fixed nugget: Gaussian likelihood with a noise variance held at 1e-8
        lik = lik_gaussian('sigma2', 1e-8, 'sigma2_prior', prior_fixed());
        % Gaussian kernel, one length-scale per input.
        % GPstuff optimizes the log-parameters: log-uniform priors cancel the Jacobian, i.e., maximum likelihood
        gpcf = gpcf_sexp('lengthScale', 0.1 * ones(1, d), 'magnSigma2', 1, ...
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
