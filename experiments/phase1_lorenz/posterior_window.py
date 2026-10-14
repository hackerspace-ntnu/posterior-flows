import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

from posterior_flows.inference import gaussian_log_likelihood
from posterior_flows.models.lorenz63 import lorenz63
from posterior_flows.observations import obs_simulation
from posterior_flows.plotting import plot_attractor
from posterior_flows.rk4 import rk4_solve


def main():
    sigma = 10
    rho = 28
    b = 8 / 3
    dt = 0.01
    t_max = 10
    obs_every = 10
    N = 2000
    rng = np.random.default_rng(0)

    # burn-in the initial guess, last point is found in the attractor
    _, Y_burn = rk4_solve(
        lorenz63, [0, t_max], np.array([1.0, 1.0, 1.0]), dt, args=(sigma, rho, b)
    )

    # burned-in initial guess
    x0 = Y_burn[:, -1]

    # solve the Lorenz differential equations and plot the solution
    t, Y = rk4_solve(lorenz63, [0, t_max], x0, dt, args=(sigma, rho, b))
    plot_attractor(t, Y, rho)

    noise_std = 1.0
    _t_obs, y_obs, _x_true = obs_simulation(
        lorenz63,
        x0,
        t_max,
        dt,
        obs_every,
        args=(sigma, rho, b),
        noise_std=noise_std,
        rng=rng,
    )

    rho_vals = np.linspace(26, 30, N)
    state0 = np.repeat(x0[:, None], N, axis=1)
    _t, X = rk4_solve(
        lorenz63,
        [0, t_max],
        state0,
        dt,
        args=(sigma, rho_vals, b),
        save_every=obs_every,
    )

    X_ens = np.moveaxis(X, -1, 0)
    loglik = np.array(
        [gaussian_log_likelihood(y_obs, x_pred, noise_std) for x_pred in X_ens]
    )
    logpost = loglik - logsumexp(loglik)  # avoid underflow

    plt.figure()
    plt.title(r"Log-likelihood for $\rho$ given data $y$")
    plt.plot(rho_vals, logpost)
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$\log p(\rho | y)$")
    plt.show()


if __name__ == "__main__":
    main()
