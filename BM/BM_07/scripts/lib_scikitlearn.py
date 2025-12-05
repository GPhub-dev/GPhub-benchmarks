"""Benchmark - scikit-learn library."""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.preprocessing import StandardScaler
import numpy as np

from gphubkit.launcher.executor import PythonGPLibrary


class SKLearnLibrary(PythonGPLibrary):
    """scikit-learn Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        # Normalize inputs
        self.scaler_x = StandardScaler()
        self._train_x = self.scaler_x.fit_transform(train_x)
        self._train_y = train_y

        # Create Kriging algorithm with Matérn 5/2 kernel and ARD
        # Added WhiteKernel for nugget estimation.
        # Since inputs are normalized, length_scale=1.0 is a good start.
        # Increased bounds to allow screening of inert variables (ARD)
        kernel = C(1.0, (1e-3, 1e3)) * Matern(
            length_scale=[1.0] * 20, length_scale_bounds=(1e-2, 1e5), nu=2.5
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,  # helps with convergence
        )

    def train(self) -> None:
        """Train the model."""
        self.gp_model.fit(self._train_x, self._train_y)

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x_norm = self.scaler_x.transform(test_x)
        pred_y, pred_std = self.gp_model.predict(test_x_norm, return_std=True)
        return pred_y, pred_std**2
