from collections.abc import Callable

import jax
import jax.numpy as jnp

from posterior_flows.models.lorenz63 import lorenz63
from posterior_flows.rk4 import rk4_solve


def gaussian_log_likelihood(
    y_obs: jnp.ndarray,
    x_pred: jnp.ndarray,
    noise_std: float | jnp.ndarray,
    include_constant: bool = False,
) -> jnp.ndarray:
    """Gaussian log-likelihood for multiple trajectories. log p(y_obs | x_pred) with i.i.d. noise.

    y_obs: (3, n_obs)
    x_pred: (..., 3, n_obs)
    noise_std: scalar, or shape (3, 1) for per-component noise
    Returns: shape (...), i.e. a scalar for a single trajectory
    """
    residual = (y_obs - x_pred) / noise_std  # (data - prediction) / noise
    ll = -0.5 * jnp.sum(residual**2, axis=(-2, -1))

    if include_constant:
        sigma = jnp.broadcast_to(noise_std, y_obs.shape)
        ll = ll - jnp.sum(jnp.log(sigma)) - 0.5 * y_obs.size * jnp.log(2 * jnp.pi)

    return ll


def make_lorenz63_log_likelihood(
    y_obs: jnp.ndarray,
    x0: jnp.ndarray,
    t_max: float,
    dt: float,
    obs_every: int,
    noise_std: float,
    sigma: float,
    b: float,
) -> Callable:
    """log p(y_obs | rho) for Lorenz-63 with known x0, sigma and b. theta = [rho]."""

    @jax.jit
    def log_likelihood(rho):
        rho = jnp.atleast_1d(jnp.asarray(rho, dtype=float))  # (N,)
        state0 = jnp.repeat(x0[:, None], rho.size, axis=1)  # (3, N)
        _, X = rk4_solve(
            lorenz63, [0, t_max], state0, dt, args=(sigma, rho, b), save_every=obs_every
        )

        X_pred = jnp.moveaxis(X, -1, 0)  # (N, 3, n_obs)

        return gaussian_log_likelihood(y_obs, X_pred, noise_std, include_constant=False)

    return log_likelihood
