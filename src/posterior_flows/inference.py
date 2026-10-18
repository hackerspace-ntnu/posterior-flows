from collections.abc import Callable

import numpy as np

from posterior_flows.models.lorenz63 import lorenz63
from posterior_flows.rk4 import rk4_solve


def gaussian_log_likelihood(
    y_obs: np.ndarray,
    x_pred: np.ndarray,
    noise_std: float | np.ndarray,
    include_constant: bool = False,
) -> np.ndarray:
    """Gaussian log-likelihood for multiple trajectories. log p(y_obs | x_pred) with i.i.d. noise.

    y_obs: (3, n_obs)
    x_pred: (..., 3, n_obs)
    noise_std: scalar, or shape (3, 1) for per-component noise
    Returns: shape (...), i.e. a scalar for a single trajectory
    """
    residual = (y_obs - x_pred) / noise_std  # (data - prediction) / noise
    ll = -0.5 * np.sum(residual**2, axis=(-2, -1))

    if include_constant:
        sigma = np.broadcast_to(noise_std, y_obs.shape)
        ll = ll - np.sum(np.log(sigma)) - 0.5 * y_obs.size * np.log(2 * np.pi)

    return ll


def make_lorenz63_log_likelihood(
    y_obs: np.ndarray,
    x0: np.ndarray,
    t_max: float,
    dt: float,
    obs_every: int,
    noise_std: float,
    sigma: float,
    b: float,
) -> Callable:
    """log p(y_obs | rho) for Lorenz-63 with known x0, sigma and b. theta = [rho]."""

    def log_likelihood(theta):
        rho = theta[0]
        _, x_pred = rk4_solve(
            lorenz63, [0, t_max], x0, dt, args=(sigma, rho, b), save_every=obs_every
        )

        return gaussian_log_likelihood(y_obs, x_pred, noise_std, include_constant=False)

    return log_likelihood
