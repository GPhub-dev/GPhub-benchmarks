"""Benchmark - GPyTorch library."""

from gpytorch.distributions import MultivariateNormal
import numpy as np
import torch
import gpytorch

from gphubkit.launcher.executor import PythonGPLibrary


class GPyTorchLibrary(PythonGPLibrary):
    """GPyTorch Library for BM_04 (Dette-Pepelyshev 3D)."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        train_x_tensor = torch.tensor(train_x, dtype=torch.float64).squeeze()
        train_y_tensor = torch.tensor(train_y, dtype=torch.float64).squeeze()
        self._train_x_tensor = train_x_tensor
        self._train_y_tensor = train_y_tensor

        class ExactGPModel(gpytorch.models.ExactGP):
            """Exact Gaussian Process model for BM_04."""

            def __init__(
                self,
                train_x: torch.Tensor,
                train_y: torch.Tensor,
                likelihood: gpytorch.likelihoods.GaussianLikelihood,
            ) -> None:
                """Initialize the GP model."""
                super(ExactGPModel, self).__init__(train_x, train_y, likelihood)  # noqa: UP008
                # Constant mean
                self.mean_module = gpytorch.means.ConstantMean()
                # Gaussian/SE Kernel with ARD (3 length scales for 3D)
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        ard_num_dims=3,
                    ),
                )

            def forward(self, x: torch.Tensor) -> MultivariateNormal:
                """Forward pass."""
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)  # type: ignore

        noise_term = 1e-8
        noise_constraint = gpytorch.constraints.GreaterThan(noise_term)  # type: ignore

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint,
            noise_prior=gpytorch.priors.NormalPrior(noise_term, noise_term * 0.1),
        )
        likelihood.initialize(noise=noise_term)

        model = ExactGPModel(train_x_tensor, train_y_tensor, likelihood)

        optimizer = torch.optim.LBFGS(
            model.parameters(),
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
        test_x_tensor = torch.tensor(test_x, dtype=torch.float64).squeeze()
        self.gp_model.eval()
        self.likelihood.eval()
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.gp_model(test_x_tensor))
            pred_y = observed_pred.mean
            pred_var = observed_pred.variance
        return pred_y.numpy(), pred_var.numpy()
