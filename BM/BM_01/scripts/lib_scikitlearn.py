"""Benchmark - scikit-learn library."""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np

from gphubkit.launcher.executor import PythonGPLibrary


class SKLearnLibrary(PythonGPLibrary):
    """scikit-learn Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        # Create Kriging algorithm with proper bounds
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1e3))
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-8,  # nugget
            normalize_y=True,  # helps with convergence
        )
        self._train_x = train_x
        self._train_y = train_y

    def train(self) -> None:
        """Train the model."""
        self.gp_model.fit(self._train_x, self._train_y)

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        pred_y, pred_std = self.gp_model.predict(test_x, return_std=True)
        return pred_y, pred_std**2
