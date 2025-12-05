"""Benchmark - SMT library."""

from smt.surrogate_models import KRG
import numpy as np

from gphubkit.launcher.executor import PythonGPLibrary


class SMTLibrary(PythonGPLibrary):
    """Python library for SMT."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """Initialize the Python library for SMT."""
        self.gp_model = KRG(
            poly="constant",
            corr="squar_exp",  # Gaussian/SE kernel
            theta0=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25],  # ARD: one value per dimension (6D)
            theta_bounds=[1e-6, 1e6],
            n_start=30,
            hyper_opt="TNC",
            nugget=1e-8,  # small fixed nugget
            random_state=42,
            print_global=False,
        )
        self.gp_model.set_training_values(train_x, train_y)

    def train(self) -> None:
        """Train the model."""
        self.gp_model.train()

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        pred_y = self.gp_model.predict_values(test_x)
        pred_var = self.gp_model.predict_variances(test_x)
        return pred_y, pred_var
