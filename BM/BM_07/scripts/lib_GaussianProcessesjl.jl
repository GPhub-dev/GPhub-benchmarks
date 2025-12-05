using GaussianProcesses
using LinearAlgebra
using Optim
using Statistics

# This script is designed to be called from Python with different ACTIONs.
# The variables train_x, train_y, test_x are passed from Python.

if ACTION == "init"
    # Normalize inputs
    global mu_x = mean(train_x, dims=1)
    global sigma_x = std(train_x, dims=1)
    sigma_x[sigma_x .== 0] .= 1.0
    train_x_norm = (train_x .- mu_x) ./ sigma_x
    
    # Normalize target
    global mu_y = mean(train_y)
    global sigma_y = std(train_y)
    if sigma_y == 0
        sigma_y = 1.0
    end
    train_y_norm = (train_y .- mu_y) ./ sigma_y

    global mConst = MeanConst(0.0)
    # Initialize kernel: Mat52Ard for Matérn 5/2 with ARD (one length scale per dimension - 20D)
    # Using normalized data, so log(1.0) = 0.0 is good initialization
    global kernel = Mat52Ard(zeros(20), 0.0)
    global logObsNoise = log(0.1)  # Start with reasonable noise (std=0.1)

    # Create GP (expects D x N inputs)
    global gp = GP(train_x_norm', vec(train_y_norm), mConst, kernel, logObsNoise)

elseif ACTION == "train"
    optimize!(gp; method=LBFGS())

elseif ACTION == "test"
    test_x_norm = (test_x .- mu_x) ./ sigma_x
    pred_y_norm, pred_var_norm = predict_y(gp, test_x_norm')
    
    global pred_y = pred_y_norm .* sigma_y .+ mu_y
    global pred_var = pred_var_norm .* (sigma_y^2)
end
