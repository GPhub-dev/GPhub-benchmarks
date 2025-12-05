"""Benchmark - OpenTURNS library."""

import numpy as np
import openturns as ot

from gphubkit.launcher.executor import PythonGPLibrary

ot.Log.Show(ot.Log.NONE)


class OpenTURNSLibrary(PythonGPLibrary):
    """OpenTURNS Library."""

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

        train_x_ot = ot.Sample(train_x_norm)
        train_y_ot = ot.Sample(train_y_norm.reshape(-1, 1))

        # Define basis and covariance model
        dim = train_x_ot.getDimension()
        basis = ot.ConstantBasisFactory(dim).build()
        scale = [1.0] * dim  # ARD: 20 length scales for 20D
        amplitude = [1.0]
        kernel = ot.MaternModel(scale, amplitude, 2.5)  # Matérn 5/2

        # Create Kriging algorithm
        self.gp_model = ot.KrigingAlgorithm(train_x_ot, train_y_ot, kernel, basis)

        # Configure optimization bounds for length scales
        # Allow wide range for ARD to detect inert dimensions
        lower_bound = [1e-2] * dim + [1e-4]  # length scales + amplitude
        upper_bound = [1e3] * dim + [1e2]   # length scales + amplitude
        scale_opt_bounds = ot.Interval(lower_bound, upper_bound)
        self.gp_model.setOptimizationBounds(scale_opt_bounds)

        # Use multi-start optimization for better hyperparameter estimation
        solver = ot.MultiStart(ot.TNC(), ot.LowDiscrepancyExperiment(
            ot.SobolSequence(), ot.ComposedDistribution([ot.Uniform(1e-2, 10.0)] * dim + [ot.Uniform(0.1, 10.0)]), 10
        ).generate())
        self.gp_model.setOptimizationAlgorithm(solver)

    def train(self) -> None:
        """Train the model."""
        self.gp_model.run()
        self.gp_model = self.gp_model.getResult()

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x_norm = (test_x - self._x_mean) / self._x_std
        test_x_ot = ot.Sample(test_x_norm)
        kriging_metamodel = self.gp_model.getMetaModel()
        pred_y_norm = np.array(kriging_metamodel(test_x_ot))
        pred_var_norm = np.array(self.gp_model.getConditionalMarginalVariance(test_x_ot))

        pred_y = pred_y_norm * self._y_std + self._y_mean
        pred_var = pred_var_norm * (self._y_std**2)
        return pred_y, pred_var
