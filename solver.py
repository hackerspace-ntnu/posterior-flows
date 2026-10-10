import numpy as np
from typing import Callable

# Lorenz-63 differential system
def lorenz63(t: float, state: np.ndarray, sigma, rho, b):
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x*y - b*z])

# RK4 step function
def rk4_step(f: Callable, t: float, s: float | np.ndarray, dt: float, *args):
    k1 = f(t, s, *args)
    k2 = f(t + dt/2, s + dt/2 * k1, *args)
    k3 = f(t + dt/2, s + dt/2 * k2, *args)
    k4 = f(t + dt, s + dt * k3, *args)

    return s + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


def rk4_solve(f: Callable, t_span: list, y0: np.ndarray, dt: float, args=(), save_every=1):
    """Fixed-step RK4. y0 can be (3,) for one trajectory or (3, N) for N at once.
    Returns t of shape (n_save,) and Y of shape (3, n_save) or (3, n_save, N)."""

    t0, t1 = t_span
    n_steps = int(round((t1 - t0) / dt))
    n_save = n_steps // save_every + 1

    y = np.array(y0, dtype=float)
    Y = np.zeros((n_save, *y.shape))
    Y[0] = y

    for n in range(1, n_steps + 1):
        y = rk4_step(f, t0 + (n - 1) * dt, y, dt, *args)
        if n % save_every == 0:
            Y[n // save_every] = y

    t = t0 + dt * save_every * np.arange(n_save)
    return t, np.moveaxis(Y, 0, 1) # time at 1. axis, like sol.y



def obs_simulation(f: Callable, x0: np.ndarray, t_max: float, dt: float, obs_every: int, args, sigma_obs: float, rng: np.random.Generator):
    """Noisy observations of the true trajectory, on the same time grid as rk4_solve."""
    t_obs, x_true = rk4_solve(f, [0, t_max], x0, dt, args=args, save_every=obs_every)
    y_obs  = x_true + sigma_obs * rng.standard_normal(x_true.shape) # noisy observation

    return t_obs, y_obs