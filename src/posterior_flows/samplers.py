from collections.abc import Callable

import numpy as np


def rwm(
    log_target: Callable,
    theta0: float | np.ndarray,
    n_steps: int,
    step_size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Random walk Metropolis. Returns samples (n_steps, d) and acceptance rate."""

    assert n_steps >= 2, "n_steps must be greater than two"

    # let starting theta be a float and an array
    theta_prev = np.array(theta0, dtype=float, ndmin=1)
    d = theta_prev.size

    # selection/samples array
    samples = np.zeros((n_steps, d))
    samples[0] = theta_prev
    # initial prior
    logp_prev = log_target(theta_prev)

    accepted_steps = 0

    for i in range(1, n_steps):
        z = rng.standard_normal(d)

        theta_prev = samples[i - 1]
        theta_next = theta_prev + step_size * z

        logp_next = log_target(theta_next)

        u = rng.random()

        if np.log(u) < logp_next - logp_prev:
            # accepted step
            samples[i] = theta_next
            logp_prev = logp_next
            accepted_steps += 1
        else:
            # denied step
            samples[i] = theta_prev

    accept_rate = accepted_steps / (n_steps - 1)

    return samples, accept_rate
