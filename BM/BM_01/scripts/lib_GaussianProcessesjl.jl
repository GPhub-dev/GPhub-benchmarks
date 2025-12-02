using GaussianProcesses
using LinearAlgebra
using Optim
using Statistics

# This script is designed to be called from Python with different ACTIONs.
# The variables train_x, train_y, test_x are passed from Python.

if ACTION == "init"
    global mZero = MeanZero()
    # Initialize kernel with appropriate length scales
    global kernel = SE(0.25, 0.01)
    global logObsNoise = -1.0

    # Create GP with default numerical jitter
    global gp = GP(train_x', vec(train_y), mZero, kernel, logObsNoise)

elseif ACTION == "train"
    optimize!(gp; method=ConjugateGradient())

elseif ACTION == "test"
    global pred_y, pred_var = predict_y(gp, test_x')
end
