"""Benchmark - GPyTorch library."""

from gpytorch.distributions import MultivariateNormal
import numpy as np
import torch
import gpytorch

from gphubkit.launcher.executor import PythonGPLibrary


class GPyTorchLibrary(PythonGPLibrary):
    """GPyTorch Library for BM_07 (Morris 20D)."""

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

        train_x_tensor = torch.tensor(train_x_norm, dtype=torch.float64).squeeze()
        train_y_tensor = torch.tensor(train_y_norm, dtype=torch.float64).squeeze()
        self._train_x_tensor = train_x_tensor
        self._train_y_tensor = train_y_tensor

        class ExactGPModel(gpytorch.models.ExactGP):
            """Exact Gaussian Process model for BM_07."""

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
                # Matérn 5/2 Kernel with ARD (20 length scales for 20D)
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(
                        nu=2.5,
                        ard_num_dims=20,
                    ),
                )

            def forward(self, x: torch.Tensor) -> MultivariateNormal:
                """Forward pass."""
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return MultivariateNormal(mean_x, covar_x)  # type: ignore

        # Allow estimation of nugget (remove strict prior)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-10)
        )
        likelihood.initialize(noise=1e-4)

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
        test_x_norm = (test_x - self._x_mean) / self._x_std
        test_x_tensor = torch.tensor(test_x_norm, dtype=torch.float64).squeeze()
        
        self.gp_model.eval()
        self.likelihood.eval()
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.gp_model(test_x_tensor))
            pred_y = observed_pred.mean.numpy() * self._y_std + self._y_mean
            pred_var = observed_pred.variance.numpy() * (self._y_std**2)
        return pred_y, pred_var
