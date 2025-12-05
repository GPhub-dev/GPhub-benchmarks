using GaussianProcesses
using LinearAlgebra
using Optim
using Statistics

# This script is designed to be called from Python with different ACTIONs.
# The variables train_x, train_y, test_x are passed from Python.

if ACTION == "init"
    global mConst = MeanConst(0.0)
    # Initialize kernel: SE(log(ℓ), log(σ)) where ℓ=length scale, σ=signal std
    global kernel = SE(log(0.1), log(1.0))
    global logObsNoise = log(1e-4)  # nugget: variance = 1e-8

    # Create GP with default numerical jitter
    global gp = GP(train_x', vec(train_y), mConst, kernel, logObsNoise)

elseif ACTION == "train"
    optimize!(gp; method=LBFGS())

elseif ACTION == "test"
    global pred_y, pred_var = predict_y(gp, test_x')
end
