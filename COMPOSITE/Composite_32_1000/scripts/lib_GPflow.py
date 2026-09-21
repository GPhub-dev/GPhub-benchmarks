"""Composite plate - GPflow (harmonized specification, see _scripts/README.md)."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import gpflow

from gphubkit.launcher.executor import PythonGPLibrary

NUGGET = 1e-8


class GPflowLibrary(PythonGPLibrary):
    """GPflow Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        m = train_x.shape[1]
        l0 = 0.25 * np.sqrt(m)
        # Gaussian kernel, one length-scale per input; constant trend, estimated
        kernel = gpflow.kernels.SquaredExponential(variance=1.0, lengthscales=[l0] * m)
        # fixed nugget: GPflow bounds the noise variance from below by 1e-6 unless told otherwise
        likelihood = gpflow.likelihoods.Gaussian(variance=NUGGET, variance_lower_bound=0.1 * NUGGET)
        self.gp_model = gpflow.models.GPR(
            data=(np.asarray(train_x, dtype=np.float64), np.asarray(train_y, dtype=np.float64).reshape(-1, 1)),
            kernel=kernel,
            mean_function=gpflow.functions.Constant(),
            likelihood=likelihood,
        )
        gpflow.set_trainable(self.gp_model.likelihood.variance, False)

    def train(self) -> None:
        """Train the model."""
        gpflow.optimizers.Scipy().minimize(  # L-BFGS-B
            self.gp_model.training_loss,
            self.gp_model.trainable_variables,
            options={"maxiter": 1000},
        )

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        pred_y, pred_var = self.gp_model.predict_y(np.asarray(test_x, dtype=np.float64))
        return pred_y.numpy(), pred_var.numpy()
