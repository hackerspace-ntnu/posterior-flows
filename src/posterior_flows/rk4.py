from collections.abc import Callable

import jax
import jax.numpy as jnp


def rk4_step(
    f: Callable, t: float, s: float | jnp.ndarray, dt: float, *args
) -> jnp.ndarray:
    """RK4 step function."""
    k1 = f(t, s, *args)
    k2 = f(t + dt / 2, s + dt / 2 * k1, *args)
    k3 = f(t + dt / 2, s + dt / 2 * k2, *args)
    k4 = f(t + dt, s + dt * k3, *args)

    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4_solve(
    f: Callable, t_span: list, y0: jnp.ndarray, dt: float, args=(), save_every: int = 1
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Fixed-step RK4. y0 can be (3,) for one trajectory or (3, N) for N at once.
    Returns t of shape (n_save,) and Y of shape (3, n_save) or (3, n_save, N)."""

    # step function to be used in lax scan
    def rk4_step_lax(y, t):
        y_new = rk4_step(f, t, y, dt, *args)
        return y_new, y_new

    t0, t1 = t_span
    n_steps = round((t1 - t0) / dt)

    # initial condition must be a float array
    y = jnp.array(y0, dtype=float)

    t = t0 + dt * jnp.arange(n_steps + 1)
    _, ys = jax.lax.scan(rk4_step_lax, y, xs=t[:-1], length=n_steps)

    # add initial condition
    Y = jnp.concatenate([jnp.expand_dims(y0, 0), ys])

    return t[::save_every], jnp.moveaxis(
        Y[::save_every], 0, 1
    )  # time at 1. axis, like sol.y
