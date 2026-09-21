"""Composite plate - GPyTorch"""

import gpytorch
import numpy as np
import torch
from gphubkit.launcher.executor import PythonGPLibrary
from gpytorch.distributions import MultivariateNormal

NUGGET = 1e-8


class GPyTorchLibrary(PythonGPLibrary):
    """GPyTorch Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        m = train_x.shape[1]
        l0 = 0.25 * np.sqrt(m)
        train_x_tensor = torch.tensor(train_x, dtype=torch.float64)
        train_y_tensor = torch.tensor(train_y, dtype=torch.float64).squeeze()
        self._train_x_tensor = train_x_tensor
        self._train_y_tensor = train_y_tensor

        class ExactGPModel(gpytorch.models.ExactGP):
            """Exact Gaussian Process model."""

            def __init__(
                self,
                train_x: torch.Tensor,
                train_y: torch.Tensor,
                likelihood: gpytorch.likelihoods.GaussianLikelihood,
            ) -> None:
                """Initialize the GP model."""
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)  # noqa: UP008
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=m)
                )
                self.covar_module.base_kernel.lengthscale = l0
                self.covar_module.outputscale = 1.0

            def forward(self, x: torch.Tensor) -> MultivariateNormal:
                """Forward pass."""
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)  # type: ignore

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(0.1 * NUGGET),  # type: ignore
        )
        likelihood.initialize(noise=NUGGET)
        likelihood.noise_covar.raw_noise.requires_grad_(False)

        model = ExactGPModel(train_x_tensor, train_y_tensor, likelihood).double()
        likelihood.double()

        optimizer = torch.optim.LBFGS(
            [p for p in model.parameters() if p.requires_grad],
            max_iter=1000,
            line_search_fn="strong_wolfe",
        )
        self.gp_model = model
        self.likelihood = likelihood
        self.optimizer = optimizer

    def train(self) -> None:
        """Train the model."""
        self.gp_model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        def closure() -> torch.Tensor:
            self.optimizer.zero_grad()
            output = self.gp_model(self._train_x_tensor)
            loss = -mll(output, self._train_y_tensor)  # type: ignore
            loss.backward()
            return loss

        self.optimizer.step(closure)

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x_tensor = torch.tensor(test_x, dtype=torch.float64)
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.gp_model(test_x_tensor))
            pred_y = observed_pred.mean
            pred_var = observed_pred.variance
        return pred_y.numpy(), pred_var.numpy()
