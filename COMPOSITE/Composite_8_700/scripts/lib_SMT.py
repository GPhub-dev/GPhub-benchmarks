"""Composite plate - SMT (harmonized specification, see _scripts/README.md)."""

from smt.surrogate_models import KRG
import numpy as np

from gphubkit.launcher.executor import PythonGPLibrary


class SMTLibrary(PythonGPLibrary):
    """Python library for SMT."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """Initialize the Python library for SMT."""
        m = train_x.shape[1]
        l0 = 0.25 * np.sqrt(m)
        # SMT: R = exp(-sum theta_k h_k^2) on inputs standardized to unit variance -> theta = var(x) / (2 l^2)
        var_x = 1.0 / 12.0
        options = {
            "poly": "constant",
            "corr": "squar_exp",
            "theta0": [var_x / (2 * l0**2)] * m,
            "theta_bounds": [var_x / (2 * 1e2**2), var_x / (2 * 1e-2**2)],  # l in [1e-2, 1e2]
            "n_start": 1,  # one COBYLA run from theta0
            "hyper_opt": "Cobyla",
            "nugget": 1e-8,
            "print_global": False,
        }
        try:
            self.gp_model = KRG(seed=42, **options)  # SMT >= 2.10
        except Exception:  # noqa: BLE001
            self.gp_model = KRG(random_state=42, **options)
        self.gp_model.set_training_values(train_x, train_y)

    def train(self) -> None:
        """Train the model."""
        self.gp_model.train()

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        pred_y = self.gp_model.predict_values(test_x)
        pred_var = self.gp_model.predict_variances(test_x)
        return pred_y, pred_var
