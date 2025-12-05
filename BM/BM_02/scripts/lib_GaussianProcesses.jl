using GaussianProcesses
using LinearAlgebra
using Optim
using Statistics

# This script is designed to be called from Python with different ACTIONs.
# The variables train_x, train_y, test_x are passed from Python.

if ACTION == "init"
    global mConst = MeanConst(0.0)
    # Initialize kernel: Mat52Ard for Matérn 5/2 with ARD (one length scale per dimension)
    global kernel = Mat52Ard([log(0.1), log(0.1)], log(1.0))
    global logObsNoise = log(1e-4)  # nugget: variance = 1e-8

    # Create GP with default numerical jitter
    global gp = GP(train_x', vec(train_y), mConst, kernel, logObsNoise)

elseif ACTION == "train"
    optimize!(gp; method=LBFGS())

elseif ACTION == "test"
    global pred_y, pred_var = predict_y(gp, test_x')
end
