from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from posterior_flows.config import Lorenz63Config
from posterior_flows.inference import make_lorenz63_log_likelihood, make_unpack_rho_x0
from posterior_flows.models.lorenz63 import lorenz63
from posterior_flows.observations import obs_simulation
from posterior_flows.rk4 import rk4_solve
from posterior_flows.samplers import rwm

jax.config.update("jax_enable_x64", True)


def main():
    cfg = Lorenz63Config()
    t_clim = 200

    n_seeds = 50
    n_steps = 20_000
    mcmc_burn = 2000
    step_size = 0.1

    # burn-in: last point lies on the attractor
    _, Y_burn = rk4_solve(
        lorenz63,
        [0, cfg.t_burn],
        jnp.array([1.0, 1.0, 1.0]),
        cfg.dt,
        args=(cfg.sigma, cfg.rho_true, cfg.b),
    )
    x0_true = Y_burn[:, -1]

    unpack = make_unpack_rho_x0(cfg.sigma, cfg.b)
    log_likelihood = make_lorenz63_log_likelihood(
        unpack, cfg.t_max, cfg.dt, cfg.obs_every, cfg.noise_std
    )

    # calculate the "climate" to draw the priors from
    _, Y_clim = rk4_solve(
        lorenz63, [0, t_clim], x0_true, cfg.dt, args=(cfg.sigma, cfg.rho_true, cfg.b)
    )
    mu_clim = np.asarray(Y_clim[:, 500:].mean(axis=1))
    Sigma_clim = np.cov(
        np.asarray(Y_clim[:, 500:])
    )  # covariance matrix for the "climate"
    Sigma_inv = np.linalg.inv(Sigma_clim)

    def log_prior(theta):
        rho, x0 = theta[0], theta[1:]
        d = x0 - mu_clim
        lp = -0.5 * d @ Sigma_inv @ d

        return jnp.where((rho >= cfg.rho_start) & (rho <= cfg.rho_end), lp, -jnp.inf)

    @jax.jit
    def log_target(theta, y_obs):
        lp = log_prior(theta)
        ll = log_likelihood(theta, y_obs)

        return jnp.where(jnp.isfinite(lp), lp + ll, -jnp.inf)

    # create random keys
    root = jax.random.key(cfg.seed)
    keys = jax.random.split(root, n_seeds)

    rho_samples, ranks, accept_rates = [], [], []

    # one independent stream per run, all derived from cfg.seed
    for i, key in enumerate(keys):
        # create random keys
        key_rho, key_x0, key_obs, key_init, key_rwm = jax.random.split(key, 5)

        rho_i = jax.random.uniform(
            key_rho, minval=cfg.rho_start, maxval=cfg.rho_end
        )  # true rho value
        x0_i = jax.random.multivariate_normal(
            key_x0, mu_clim, Sigma_clim
        )  # true start position

        _, y_obs, _ = obs_simulation(
            lorenz63,
            x0_i,
            cfg.t_max,
            cfg.dt,
            cfg.obs_every,
            args=(cfg.sigma, rho_i, cfg.b),
            noise_std=cfg.noise_std,
            key=key_obs,
        )

        theta0 = np.concatenate(
            [
                [
                    jax.random.uniform(
                        key_init, minval=cfg.rho_start, maxval=cfg.rho_end
                    )
                ],
                y_obs[:, 0],
            ]
        )
        samples, accept_rate = rwm(
            partial(log_target, y_obs=y_obs),
            theta0=theta0,
            n_steps=n_steps,
            step_size=step_size,
            key=key_rwm,
        )

        rho_samples.append(samples[mcmc_burn:, 0])

        theta_i = np.concatenate([[rho_i], x0_i])
        ranks.append(np.mean(samples[mcmc_burn:] < theta_i, axis=0))
        accept_rates.append(accept_rate)

    ranks = np.array(ranks)
    # rank in [0.025, 0.975] iff theta_i inside the central 95% interval
    coverage = np.mean((ranks >= 0.025) & (ranks <= 0.975))

    print(f"95% coverage: {coverage:.2f}")
    print(f"accept rate: min {min(accept_rates):.2f}, max {max(accept_rates):.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(np.concatenate(rho_samples), bins=50, density=True)
    ax1.set_xlabel(r"$\rho$")
    ax1.set_title(f"Pooled posterior ({n_seeds} seeds)")

    # histogram for ranks for the rho samples
    ax2.hist(ranks[:, 0], bins=10, range=(0, 1))
    ax2.axhline(
        n_seeds / 10, color="k", ls="--"
    )  # expected count per bin if calibrated
    ax2.set_xlabel(r"rank of $\rho_{true}$")
    ax2.set_title(f"Ranks (95% coverage: {coverage:.2f})")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
