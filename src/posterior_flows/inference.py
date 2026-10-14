import numpy as np


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
