% Composite plate - GPML (harmonized specification, see _scripts/README.md)

switch ACTION
    case 'init'
        % INITIALIZE GPML
        addpath('~/Documents/MATLAB/gpml-matlab-v4.2-2018-06-11')
        startup
        m = size(train_x, 2);
        l0 = 0.25 * sqrt(m);
        meanfunc = @meanConst;           % constant trend, estimated
        covfunc = @covSEard;             % Gaussian kernel, one length-scale per input
        likfunc = @likGauss;             % Gaussian likelihood
        % GPML hyperparameters are logarithms
        hyp = struct();
        hyp.mean = 0;
        hyp.cov = [log(l0) * ones(m, 1); log(1)];   % log length-scales, log signal standard deviation
        hyp.lik = log(sqrt(1e-8));                  % log noise standard deviation (nugget 1e-8)
        % fixed nugget: the noise hyperparameter is excluded from the optimization
        prior = struct();
        prior.lik = {{@priorDelta}};
        infmethod = {@infPrior, @infGaussLik, prior};
    case 'train'
        % TRAINING (conjugate gradients, at most 1000 function evaluations)
        hyp_opt = minimize(hyp, @gp, -1000, infmethod, meanfunc, covfunc, likfunc, train_x, train_y);
    case 'test'
        % PREDICTION
        [pred_y, pred_var] = gp(hyp_opt, infmethod, meanfunc, covfunc, likfunc, train_x, train_y, test_x);
end
