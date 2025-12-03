"""Benchmark - GPJax library."""

from jax import jit, config
import gpjax as gpx
import numpy as np
import jax.numpy as jnp

from gphubkit.launcher.executor import PythonGPLibrary

config.update("jax_enable_x64", True)


class SKLearnLibrary(PythonGPLibrary):
    """scikit-learn Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        # Convert to jax arrays
        train_x_jnp = jnp.array(train_x)
        train_y_jnp = jnp.array(train_y).reshape(-1, 1)

        self.D = gpx.Dataset(X=train_x_jnp, y=train_y_jnp)

        # Define kernel and mean
        kernel = gpx.kernels.RBF(
            lengthscale=0.25,
        )
        mean = gpx.mean_functions.Zero()

        # Prior
        prior = gpx.gps.Prior(kernel=kernel, mean_function=mean)

        # Likelihood with matched nugget/alpha term
        likelihood = gpx.likelihoods.Gaussian(
            num_datapoints=self.D.n,
        )
        # Posterior
        self.posterior = prior * likelihood

    def train(self) -> None:
        """Train the model."""
        self.opt_posterior, _ = gpx.fit_scipy(
            model=self.posterior,
            objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),  # type: ignore
            train_data=self.D,
            max_iters=1000,  # Match n_restarts_optimizer
            verbose=False,
        )

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x_jnp = jnp.array(test_x)
        batch_size = 1000
        n_test = len(test_x_jnp)

        pred_means = []
        pred_stds = []

        @jit
        def predict_fn(x: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            latent_dist = self.opt_posterior.predict(x, train_data=self.D)
            predictive_dist = self.opt_posterior.likelihood(latent_dist)
            return predictive_dist.mean(), predictive_dist.stddev()

        for i in range(0, n_test, batch_size):
            batch_x = test_x_jnp[i : min(i + batch_size, n_test)]
            batch_mean, batch_std = predict_fn(batch_x)
            pred_means.append(np.array(batch_mean))
            pred_stds.append(np.array(batch_std))

        pred_y = np.concatenate(pred_means)
        pred_var = np.concatenate(pred_stds)
        return pred_y, pred_var
