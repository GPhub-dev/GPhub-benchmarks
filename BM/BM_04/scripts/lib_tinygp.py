"""Benchmark BM_04 (function 3, Dette & Pepelyshev) - tinygp library."""

from scipy.optimize import minimize
import jax
import jax.flatten_util
import numpy as np
import tinygp
import jax.numpy as jnp

from gphubkit.launcher.executor import PythonGPLibrary

jax.config.update("jax_enable_x64", True)


def build_gp(params: dict, x: jnp.ndarray) -> tinygp.GaussianProcess:
    """Constant trend; Gaussian kernel with one length-scale per input."""
    kernel = jnp.exp(params["log_variance"]) * tinygp.transforms.Linear(
        jnp.exp(-params["log_lengthscale"]), tinygp.kernels.ExpSquared()
    )
    return tinygp.GaussianProcess(kernel, x, diag=1e-8, mean=params["mean"])


class TinyGPLibrary(PythonGPLibrary):
    """tinygp Library."""

    def __init__(self, train_x: np.ndarray, train_y: np.ndarray) -> None:
        """GPR model."""
        self._x_mean, self._x_std, self._y_mean, self._y_std = 0.0, 1.0, 0.0, 1.0  # data already standardized
        d = train_x.shape[1]
        self._x = jnp.asarray(train_x)
        self._y = jnp.asarray(train_y).ravel()
        self.params = {
            "mean": jnp.zeros(()),
            "log_variance": jnp.zeros(()),
            "log_lengthscale": jnp.full(d, jnp.log(0.1)),
        }

    def train(self) -> None:
        """Train the model: maximum likelihood with L-BFGS-B (tinygp ships no optimizer)."""
        flat, unravel = jax.flatten_util.ravel_pytree(self.params)

        @jax.jit
        def loss(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            return jax.value_and_grad(lambda t: -build_gp(unravel(t), self._x).log_probability(self._y))(theta)

        def fun(theta: np.ndarray) -> tuple[float, np.ndarray]:
            value, grad = loss(jnp.asarray(theta))
            return float(value), np.asarray(grad, dtype=np.float64)

        res = minimize(fun, np.asarray(flat), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
        self.params = unravel(jnp.asarray(res.x))

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the values and variances of the model."""
        gp = build_gp(self.params, self._x)
        test_x_jnp = jnp.asarray((test_x - self._x_mean) / self._x_std)
        diag = 1e-8  # predictive variance of the observations

        @jax.jit
        def predict_fn(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            cond = gp.condition(self._y, x, diag=diag).gp
            return cond.loc, cond.variance

        means, variances = [], []
        for i in range(0, len(test_x_jnp), 1000):
            mean, var = predict_fn(test_x_jnp[i : i + 1000])
            means.append(np.asarray(mean))
            variances.append(np.asarray(var))
        pred_y = np.concatenate(means) * self._y_std + self._y_mean
        pred_var = np.concatenate(variances) * self._y_std**2
        return pred_y, pred_var
