import numpy as np

from plot import plot_attractor
from solver import lorenz63, rk4_solve, obs_simulation


def main():
    sigma = 10; rho = 28; b = 8/3
    dt = 0.01; t_max = 10; obs_every = 10
    N = 2000
    rng = np.random.default_rng(0)

    _, Y_burn = rk4_solve(lorenz63, [0, t_max], np.array([1.0, 1.0, 1.0]), dt, args=(sigma, rho, b))

    x0 = Y_burn[:, -1]

    t, Y = rk4_solve(lorenz63, [0, t_max], x0, dt, args=(sigma, rho, b))
    plot_attractor(t, Y, rho)

    t_obs, y_obs = obs_simulation(lorenz63, x0, t_max, dt, obs_every, args=(sigma, rho, b), sigma_obs=1.0, rng=rng)

    rho_vals = np.linspace(26, 30, N)
    state0 = np.repeat(x0[:, None], N, axis=1)
    t_vals, X = rk4_solve(lorenz63, [0, t_max], state0, dt, args=(sigma, rho_vals, b), save_every=obs_every)



if __name__ == "__main__":
    main()
