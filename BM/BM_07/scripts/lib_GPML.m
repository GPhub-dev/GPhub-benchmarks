% GPML

switch ACTION
    case 'init'
        % INITIALIZE GPML
        addpath('~/Documents/MATLAB/gpml-matlab-v4.2-2018-06-11')
        startup
        
        % Normalize inputs
        mu_x = mean(train_x);
        sigma_x = std(train_x);
        sigma_x(sigma_x==0) = 1;
        train_x = (train_x - mu_x) ./ sigma_x;
        
        % Normalize target
        mu_y = mean(train_y);
        sigma_y = std(train_y);
        if sigma_y==0, sigma_y=1; end
        train_y = (train_y - mu_y) ./ sigma_y;

        meanfunc = @meanConst;           % Constant mean function
        covfunc = {@covMaternard, 5};    % Matérn 5/2 kernel with ARD
        likfunc = @likGauss;             % Gaussian likelihood
        hyp = struct();
        hyp.mean = 0;                    % Initial value for constant mean function
        % Length scales: log(1.0) = 0.0 since normalized. + Signal var log(1.0)=0.0
        % 20 dimensions -> 20 length scales + 1 signal variance = 21 parameters
        hyp.cov = zeros(21,1); 
        hyp.lik = log(0.01);             % Initial noise std dev (0.01 -> 1e-4 variance)
    case 'train'
        % TRAINING
        hyp_opt = minimize(hyp, @gp, -200, @infGaussLik, meanfunc, covfunc, likfunc, train_x, train_y);
    case 'test'
        % PREDICTION
        test_x_norm = (test_x - mu_x) ./ sigma_x;
        [pred_y_norm, pred_var_norm] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, train_x, train_y, test_x_norm);
        pred_y = pred_y_norm * sigma_y + mu_y;
        pred_var = pred_var_norm * sigma_y^2;
end
