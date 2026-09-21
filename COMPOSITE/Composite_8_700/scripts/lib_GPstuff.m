% Composite plate - GPstuff (harmonized specification, see _scripts/README.md)

switch ACTION
    case 'init'
        % INITIALIZE GPstuff
        gpstuff_root = '~/Documents/MATLAB/gpstuff-develop';
        for sf = {'diag', 'dist', 'gp', 'mc', 'misc', 'optim'}
            addpath(fullfile(gpstuff_root, sf{:}))
        end
        m = size(train_x, 2);
        l0 = 0.25 * sqrt(m);
        % fixed nugget: Gaussian likelihood with a noise variance held at 1e-8
        lik = lik_gaussian('sigma2', 1e-8, 'sigma2_prior', prior_fixed());
        % Gaussian kernel, one length-scale per input.
        % GPstuff optimizes the log-parameters: log-uniform priors cancel the Jacobian, i.e., maximum likelihood
        gpcf = gpcf_sexp('lengthScale', l0 * ones(1, m), 'magnSigma2', 1, ...
            'lengthScale_prior', prior_logunif(), 'magnSigma2_prior', prior_logunif());
        % constant trend: constant basis function whose weight has a vague Gaussian prior
        gpmf = gpmf_constant('prior_mean', 0, 'prior_cov', 100);
        gp = gp_set('lik', lik, 'cf', {gpcf}, 'meanf', {gpmf}, 'jitterSigma2', 0);
        opt = optimset('TolFun', 1e-6, 'TolX', 1e-6, 'MaxIter', 1000, 'Display', 'off');
    case 'train'
        % TRAINING (scaled conjugate gradients, the default optimizer of GPstuff)
        gp = gp_optim(gp, train_x, train_y, 'opt', opt);
    case 'test'
        % PREDICTION
        [pred_y, pred_var_f] = gp_pred(gp, train_x, train_y, test_x);
        pred_var = pred_var_f + gp.lik.sigma2;   % predictive variance of the observations
end
