% Composite plate - GPML

switch ACTION
    case 'init'
        addpath('~/Documents/MATLAB/gpml-matlab-v4.2-2018-06-11')
        startup
        m = size(train_x, 2);
        l0 = 0.25 * sqrt(m);
        meanfunc = @meanConst;
        covfunc = @covSEard;
        likfunc = @likGauss;
        hyp = struct();
        hyp.mean = 0;
        hyp.cov = [log(l0) * ones(m, 1); log(1)];
        hyp.lik = log(sqrt(1e-8));
        prior = struct();
        prior.lik = {{@priorDelta}};
        infmethod = {@infPrior, @infGaussLik, prior};
    case 'train'
        hyp_opt = minimize(hyp, @gp, -1000, infmethod, meanfunc, covfunc, likfunc, train_x, train_y);
    case 'test'
        [pred_y, pred_var] = gp(hyp_opt, infmethod, meanfunc, covfunc, likfunc, train_x, train_y, test_x);
end
