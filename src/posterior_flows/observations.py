from collections.abc import Callable

import jax

from posterior_flows.rk4 import rk4_solve


def obs_simulation(
    f: Callable,
    x0: jax.Array,
    t_max: float,
    dt: float,
    obs_every: int,
    args,
    noise_std: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Noisy observations of the true trajectory, on the same time grid as rk4_solve."""
    t_obs, x_true = rk4_solve(f, [0, t_max], x0, dt, args=args, save_every=obs_every)
    y_obs = x_true + noise_std * jax.random.normal(
        key, x_true.shape, dtype=x_true.dtype
    )  # noisy observation

    return t_obs, y_obs, x_true
