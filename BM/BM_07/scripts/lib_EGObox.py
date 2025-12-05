"""Benchmark - EGObox library."""

import numpy as np
import egobox as egx

from gphubkit.launcher.executor import PythonGPLibrary


class EGOboxLibrary(PythonGPLibrary):
    """EGObox Library."""

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

        # Matérn 5/2 kernel with constant mean and improved optimization
        self.gp_model = egx.Gpx.builder(  # type: ignore
            regr_spec=egx.RegressionSpec.CONSTANT,  # type: ignore
            corr_spec=egx.CorrelationSpec.MATERN52,  # type: ignore
            n_start=20,  # More restarts for better optimization
            seed=42,  # Reproducibility
        )
        self._train_x = train_x_norm
        self._train_y = train_y_norm

    def train(self) -> None:
        """Train the model."""
        self.gp_model = self.gp_model.fit(
            self._train_x,
            self._train_y.reshape(-1, 1),
        )  # type: ignore

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x_norm = (test_x - self._x_mean) / self._x_std
        pred_y_norm = self.gp_model.predict(test_x_norm)
        pred_var_norm = self.gp_model.predict_var(test_x_norm)

        pred_y = pred_y_norm * self._y_std + self._y_mean
        pred_var = pred_var_norm * (self._y_std**2)
        return pred_y, pred_var
