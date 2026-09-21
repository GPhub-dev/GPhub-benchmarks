"""Benchmark BM_06 (function 5, borehole) - GPflow library."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import gpflow

from gphubkit.launcher.executor import PythonGPLibrary


class GPflowLibrary(PythonGPLibrary):
    """GPflow Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        # Normalize inputs and target with the training statistics (data stored in physical units)
        self._x_mean, self._x_std = train_x.mean(axis=0), train_x.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        self._y_mean, self._y_std = train_y.mean(), train_y.std()
        train_x = (train_x - self._x_mean) / self._x_std
        train_y = (train_y - self._y_mean) / self._y_std
        d = train_x.shape[1]
        # Matern 5/2 kernel, one length-scale per input; constant trend, estimated
        kernel = gpflow.kernels.Matern52(variance=1.0, lengthscales=[1.0] * d)
        # noise variance estimated, initial value 1e-2
        likelihood = gpflow.likelihoods.Gaussian(variance=1e-2)
        self.gp_model = gpflow.models.GPR(
            data=(np.asarray(train_x, dtype=np.float64), np.asarray(train_y, dtype=np.float64).reshape(-1, 1)),
            kernel=kernel,
            mean_function=gpflow.functions.Constant(),
            likelihood=likelihood,
        )

    def train(self) -> None:
        """Train the model."""
        gpflow.optimizers.Scipy().minimize(  # L-BFGS-B
            self.gp_model.training_loss,
            self.gp_model.trainable_variables,
            options={"maxiter": 1000},
        )

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x = (np.asarray(test_x, dtype=np.float64) - self._x_mean) / self._x_std
        pred_y, pred_var = self.gp_model.predict_y(test_x)
        return pred_y.numpy() * self._y_std + self._y_mean, pred_var.numpy() * self._y_std**2
