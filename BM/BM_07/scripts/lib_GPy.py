"""Benchmark - GPy library."""

from GPy.models import GPRegression as GPyGPRegression
import GPy
import numpy as np

from gphubkit.launcher.executor import PythonGPLibrary


class GPyLibrary(PythonGPLibrary):
    """GPy Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        # Normalize inputs
        self._x_mean = train_x.mean(axis=0)
        self._x_std = train_x.std(axis=0)
        self._x_std[self._x_std == 0] = 1.0
        train_x_norm = (train_x - self._x_mean) / self._x_std

        # Normalize target
        self._y_mean = train_y.mean()
        self._y_std = train_y.std()
        if self._y_std == 0:
            self._y_std = 1.0
        train_y_norm = (train_y - self._y_mean) / self._y_std

        # Matérn 5/2 kernel with ARD (20 length scales for 20D Morris)
        kernel = GPy.kern.Matern52(input_dim=20, ARD=True)
        # Initialize GPR without fixed noise (let it estimate)
        gp_gpy = GPyGPRegression(train_x_norm, train_y_norm.reshape(-1, 1), kernel=kernel)
        # Initialize likelihood variance to a reasonable value
        gp_gpy.likelihood.variance = 1e-2

        gp_gpy.optimize_restarts(
            num_restarts=10,
            optimizer="bfgs",
            robust=True,
            verbose=False,
        )
        self.gp_model = gp_gpy

    def train(self) -> None:
        """Train the model."""
        # Already optimized in __init__ via optimize_restarts
        pass

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x_norm = (test_x - self._x_mean) / self._x_std
        pred_y_norm, pred_var_norm = self.gp_model.predict(test_x_norm)
        
        pred_y = pred_y_norm * self._y_std + self._y_mean
        pred_var = pred_var_norm * (self._y_std**2)
        return pred_y, pred_var
