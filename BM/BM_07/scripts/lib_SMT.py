"""Benchmark - SMT library."""

from smt.surrogate_models import KRG
import numpy as np

from gphubkit.launcher.executor import PythonGPLibrary


class SMTLibrary(PythonGPLibrary):
    """Python library for SMT."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """Initialize the Python library for SMT."""
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

        self.gp_model = KRG(
            poly="constant",
            corr="matern52",  # Matérn 5/2 kernel
            theta0=[1.0] * 20,  # ARD: one value per dimension (20D), normalized
            theta_bounds=[1e-3, 1e4],  # Wider bounds for length scales
            n_start=20,  # More restarts for better optimization
            hyper_opt="Cobyla",  # TNC often works better than Cobyla
            nugget=1e-6,  # Smaller nugget for better fit
            seed=42,  # named 'random_state' in the SMT releases of the first campaign
            print_global=False,
        )
        self.gp_model.set_training_values(train_x_norm, train_y_norm)

    def train(self) -> None:
        """Train the model."""
        self.gp_model.train()

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x_norm = (test_x - self._x_mean) / self._x_std
        pred_y_norm = self.gp_model.predict_values(test_x_norm)
        pred_var_norm = self.gp_model.predict_variances(test_x_norm)

        pred_y = pred_y_norm * self._y_std + self._y_mean
        pred_var = pred_var_norm * (self._y_std**2)
        return pred_y, pred_var
