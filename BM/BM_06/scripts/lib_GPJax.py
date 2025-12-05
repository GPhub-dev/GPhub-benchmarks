"""Benchmark - GPJax library."""

from jax import jit, config
import gpjax as gpx
import numpy as np
import jax.numpy as jnp

from gphubkit.launcher.executor import PythonGPLibrary

config.update("jax_enable_x64", True)


class GPJaxLibrary(PythonGPLibrary):
    """GPJax Library."""

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

        # Convert to jax arrays
        train_x_jnp = jnp.array(train_x_norm)
        train_y_jnp = jnp.array(train_y_norm).reshape(-1, 1)

        self.D = gpx.Dataset(X=train_x_jnp, y=train_y_jnp)

        # Define kernel and mean - Matérn 5/2 with ARD (8 length scales for 8D Borehole)
        # Since inputs are normalized, initialize lengthscale to 1.0
        kernel = gpx.kernels.Matern52(
            lengthscale=jnp.array([1.0] * 8),
        )
        mean = gpx.mean_functions.Constant()

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
            max_iters=500,
            verbose=False,
        )

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        test_x_norm = (test_x - self._x_mean) / self._x_std
        test_x_jnp = jnp.array(test_x_norm)
        batch_size = 1000
        n_test = len(test_x_jnp)

        pred_means = []
        pred_vars = []

        @jit
        def predict_fn(x: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            latent_dist = self.opt_posterior.predict(x, train_data=self.D)
            predictive_dist = self.opt_posterior.likelihood(latent_dist)
            return predictive_dist.mean(), predictive_dist.variance()

        for i in range(0, n_test, batch_size):
            batch_x = test_x_jnp[i : min(i + batch_size, n_test)]
            batch_mean, batch_var = predict_fn(batch_x)
            pred_means.append(np.array(batch_mean))
            pred_vars.append(np.array(batch_var))

        pred_y = np.concatenate(pred_means) * self._y_std + self._y_mean
        pred_var = np.concatenate(pred_vars) * (self._y_std**2)
        return pred_y, pred_var
