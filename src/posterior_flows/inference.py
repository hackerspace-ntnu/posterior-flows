from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from posterior_flows.models.lorenz63 import lorenz63
from posterior_flows.rk4 import rk4_solve


def gaussian_log_likelihood(
    y_obs: jax.Array,
    x_pred: jax.Array,
    noise_std: float | jax.Array,
    include_constant: bool = False,
) -> jax.Array:
    """Gaussian log-likelihood for multiple trajectories. log p(y_obs | x_pred) with i.i.d. noise.

    y_obs: (d, n_obs)
    x_pred: (..., d, n_obs)
    noise_std: scalar, or shape (d, 1) for per-component noise
    Returns: shape (...), i.e. a scalar for a single trajectory
    """
    residual = (y_obs - x_pred) / noise_std  # (data - prediction) / noise
    ll = -0.5 * jnp.sum(residual**2, axis=(-2, -1))

    if include_constant:
        sigma = jnp.broadcast_to(noise_std, y_obs.shape)
        ll = ll - jnp.sum(jnp.log(sigma)) - 0.5 * y_obs.size * jnp.log(2 * jnp.pi)

    return ll


def make_unpack_rho_x0(sigma: float, b: float) -> Callable:
    """theta = [rho, x0_x, x0_y, x0_z]."""

    def unpack(theta: jax.Array):
        return theta[:, 1:].T, (sigma, theta[:, 0], b)

    return unpack


def make_lorenz63_log_likelihood(
    unpack: Callable, t_max: float, dt: float, obs_every: int, noise_std: float
) -> Callable:
    """log p(y_obs | theta). theta with shape (d,) gives a scalar and shape (N, d) gives (N,)."""

    @jax.jit
    def log_likelihood(
        theta: ArrayLike, y_obs: jax.Array
    ) -> jax.Array:  # y_obs as argument, so it is compiled once
        theta = jnp.atleast_2d(jnp.asarray(theta, dtype=theta.dtype))
        state0, args = unpack(theta)
        _, X = rk4_solve(
            lorenz63, [0, t_max], state0, dt, args=args, save_every=obs_every
        )
        X_pred = jnp.moveaxis(X, -1, 0)  # (N, d, n_obs)
        ll = gaussian_log_likelihood(y_obs, X_pred, noise_std, include_constant=False)

        return ll[0] if theta.ndim == 1 else ll

    return log_likelihood
