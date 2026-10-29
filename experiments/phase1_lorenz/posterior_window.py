from dataclasses import replace

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from posterior_flows.config import Lorenz63Config
from posterior_flows.inference import make_lorenz63_log_likelihood, make_unpack_rho_x0
from posterior_flows.models.lorenz63 import lorenz63
from posterior_flows.observations import obs_simulation
from posterior_flows.plotting import plot_attractor
from posterior_flows.rk4 import rk4_solve

jax.config.update("jax_enable_x64", True)


def main():
    cfg = replace(Lorenz63Config(), t_max=5.0)
    args = (cfg.sigma, cfg.rho_true, cfg.b)
    N = 2000
    rng = np.random.default_rng(cfg.seed)

    # burn-in the initial guess, last point is found in the attractor
    _, Y_burn = rk4_solve(
        lorenz63, [0, cfg.t_burn], np.array([1.0, 1.0, 1.0]), cfg.dt, args=args
    )

    # burned-in initial guess
    x0 = Y_burn[:, -1]

    # solve the Lorenz differential equations and plot the solution
    t, Y = rk4_solve(lorenz63, [0, cfg.t_max], x0, cfg.dt, args=args)
    plot_attractor(t, Y, cfg.rho_true)

    _t_obs, y_obs, _x_true = obs_simulation(
        lorenz63,
        x0,
        cfg.t_max,
        cfg.dt,
        cfg.obs_every,
        args=args,
        noise_std=cfg.noise_std,
        rng=rng,
    )

    rho_vals = np.linspace(cfg.rho_start, cfg.rho_end, N)

    unpack = make_unpack_rho_x0(cfg.sigma, cfg.b)
    log_likelihood = make_lorenz63_log_likelihood(
        unpack, cfg.t_max, cfg.dt, cfg.obs_every, cfg.noise_std
    )

    theta = np.column_stack([rho_vals, np.tile(x0, (N, 1))])  # (N, 4)
    loglik = log_likelihood(theta, y_obs)
    logpost = loglik - logsumexp(loglik)  # avoid underflow

    plt.figure()
    plt.title(r"Log-likelihood for $\rho$ given data $y$")
    plt.plot(rho_vals, logpost)
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$\log p(\rho | y)$")
    plt.show()


if __name__ == "__main__":
    main()
