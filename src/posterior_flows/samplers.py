from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def rwm(
    log_target: Callable,
    theta0: ArrayLike,
    n_steps: int,
    step_size: float,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Random walk Metropolis. Returns samples (n_steps, d) and acceptance rate."""

    assert n_steps >= 2, "n_steps must be at least 2"

    # let starting theta be a float or an array
    theta0 = jnp.array(theta0, dtype=float, ndmin=1)
    logp0 = log_target(theta0)

    # each iteration
    def step(carry, key_i):
        theta, logp = carry
        key_prop, key_acc = jax.random.split(key_i)

        # proposal
        z = jax.random.normal(key_prop, theta.shape, dtype=theta.dtype)
        theta_prop = theta + step_size * z
        logp_prop = log_target(theta_prop)

        # accept/reject
        accept = jnp.log(jax.random.uniform(key_acc)) < logp_prop - logp
        theta = jnp.where(accept, theta_prop, theta)
        logp = jnp.where(accept, logp_prop, logp)

        return (theta, logp), (theta, accept)  # new carry, return data

    keys = jax.random.split(key, n_steps - 1)  # one key per step
    _, (thetas, accepted) = jax.lax.scan(step, (theta0, logp0), keys)

    samples = jnp.concatenate([theta0[None], thetas])
    accept_rate = jnp.mean(accepted)

    return samples, accept_rate
