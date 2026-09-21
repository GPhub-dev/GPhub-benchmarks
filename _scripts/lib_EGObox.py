"""Composite plate - egobox-gp"""

import egobox as egx
import numpy as np
from gphubkit.launcher.executor import PythonGPLibrary


class EGOboxLibrary(PythonGPLibrary):
    """EGObox Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        m = train_x.shape[1]
        l0 = 0.25 * np.sqrt(m)
        var_x = 1.0 / 12.0
        self.gp_model = egx.Gpx.builder(  # type: ignore
            regr_spec=egx.RegressionSpec.CONSTANT,  # type: ignore
            corr_spec=egx.CorrelationSpec.SQUARED_EXPONENTIAL,  # type: ignore
            theta_init=[var_x / (2 * l0**2)] * m,
            theta_bounds=[[var_x / (2 * 1e2**2), var_x / (2 * 1e-2**2)]] * m,
            n_start=1,
            seed=42,
        )
        self._train_x = train_x
        self._train_y = train_y

    def train(self) -> None:
        """Train the model."""
        self.gp_model = self.gp_model.fit(
            self._train_x,
            self._train_y.reshape(-1, 1),
        )  # type: ignore

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        pred_y = self.gp_model.predict(test_x)
        pred_var = self.gp_model.predict_var(test_x)
        return pred_y, pred_var
