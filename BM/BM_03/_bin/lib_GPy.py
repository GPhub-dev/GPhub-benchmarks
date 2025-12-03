"""Benchmark - GPy library."""

from GPy.models import GPRegression as GPyGPRegression
import GPy
import numpy as np

from gphubkit.launcher.executor import PythonGPLibrary


class GPyLibrary(PythonGPLibrary):
    """GPy Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        kernel = GPy.kern.RBF(input_dim=2)
        gp_gpy = GPyGPRegression(train_x, train_y.reshape(-1, 1), kernel=kernel, noise_var=1e-5)
        gp_gpy.optimize_restarts(
            num_restarts=30,
            optimizer="bfgs",
            robust=True,
            verbose=False,
        )
        self.gp_model = gp_gpy

    def train(self) -> None:
        """Train the model."""
        self.gp_model.optimize()

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        pred_y, pred_var = self.gp_model.predict(test_x)
        return pred_y, pred_var
