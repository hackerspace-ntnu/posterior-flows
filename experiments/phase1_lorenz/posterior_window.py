import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from posterior_flows.inference import make_lorenz63_log_likelihood, make_unpack_rho_x0
from posterior_flows.models.lorenz63 import lorenz63
from posterior_flows.observations import obs_simulation
from posterior_flows.plotting import plot_attractor
from posterior_flows.rk4 import rk4_solve

jax.config.update("jax_enable_x64", True)


def main():
    sigma = 10
    rho_start = 20
    rho_end = 35
    rho_true = 28
    b = 8 / 3
    dt = 0.01
    t_max = 5
    obs_every = 10
    N = 2000
    rng = np.random.default_rng(0)

    # burn-in the initial guess, last point is found in the attractor
    _, Y_burn = rk4_solve(
        lorenz63, [0, t_max], np.array([1.0, 1.0, 1.0]), dt, args=(sigma, rho_true, b)
    )

    # burned-in initial guess
    x0 = Y_burn[:, -1]

    # solve the Lorenz differential equations and plot the solution
    t, Y = rk4_solve(lorenz63, [0, t_max], x0, dt, args=(sigma, rho_true, b))
    plot_attractor(t, Y, rho_true)

    noise_std = 1.0
    _t_obs, y_obs, _x_true = obs_simulation(
        lorenz63,
        x0,
        t_max,
        dt,
        obs_every,
        args=(sigma, rho_true, b),
        noise_std=noise_std,
        rng=rng,
    )

    rho_vals = np.linspace(rho_start, rho_end, N)

    unpack = make_unpack_rho_x0(sigma, b)
    log_likelihood = make_lorenz63_log_likelihood(
        unpack, t_max, dt, obs_every, noise_std
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
