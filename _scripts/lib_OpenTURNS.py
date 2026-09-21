"""Composite plate - OpenTURNS"""

import numpy as np
import openturns as ot
from gphubkit.launcher.executor import PythonGPLibrary

ot.Log.Show(ot.Log.NONE)


class OpenTURNSLibrary(PythonGPLibrary):
    """OpenTURNS Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        m = train_x.shape[1]
        l0 = 0.25 * np.sqrt(m)
        train_x_ot = ot.Sample(train_x)
        train_y_ot = ot.Sample(train_y.reshape(-1, 1))

        basis = ot.ConstantBasisFactory(m).build()
        kernel = ot.SquaredExponential([l0] * m, [1.0])
        kernel.setNuggetFactor(1e-8)
        self.gp_model = ot.KrigingAlgorithm(train_x_ot, train_y_ot, kernel, basis)

        self.gp_model.setOptimizationBounds(ot.Interval([1e-2] * m, [1e2] * m))
        solver = ot.TNC()
        if hasattr(solver, "setCheckStatus"):
            solver.setCheckStatus(False)
        self.gp_model.setOptimizationAlgorithm(solver)

    def train(self) -> None:
        """Train the model."""
        self.gp_model.run()
        self.gp_model = self.gp_model.getResult()

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x_ot = ot.Sample(test_x)
        kriging_metamodel = self.gp_model.getMetaModel()
        pred_y = np.array(kriging_metamodel(test_x_ot))
        pred_var = np.array(self.gp_model.getConditionalMarginalVariance(test_x_ot))
        return pred_y, pred_var
