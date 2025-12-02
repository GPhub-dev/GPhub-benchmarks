"""Benchmark - GPJax library."""

from jax import config
import gpjax as gpx
import numpy as np
import jax.numpy as jnp

from gphubkit.launcher.executor import PythonGPLibrary

config.update("jax_enable_x64", True)


class GPJaxLibrary(PythonGPLibrary):
    """GPJax Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        # Convert to jax arrays
        train_x_jnp = jnp.array(train_x)
        train_y_jnp = jnp.array(train_y).reshape(-1, 1)

        self.D = gpx.Dataset(X=train_x_jnp, y=train_y_jnp)

        # Define kernel and mean - kernel params will be optimized
        kernel = gpx.kernels.RBF()
        mean = gpx.mean_functions.Constant()

        # Prior
        prior = gpx.gps.Prior(kernel=kernel, mean_function=mean)

        # Likelihood with small noise for numerical stability
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=self.D.n,
            obs_stddev=jnp.array(1e-5),
        )
        # Posterior
        self.posterior = prior * likelihood

    def train(self) -> None:
        """Train the model."""
        self.opt_posterior, _ = gpx.fit_scipy(
            model=self.posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),  # type: ignore
            train_data=self.D,
            max_iters=1000,
            verbose=False,
        )

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x_jnp = jnp.array(test_x)

        # Use JIT-compiled prediction function
        latent_dist = self.opt_posterior.predict(test_x_jnp, train_data=self.D)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)

        pred_y = np.array(predictive_dist.mean()).flatten()
        pred_var = np.array(predictive_dist.variance()).flatten()
        return pred_y, pred_var
