import matplotlib.pyplot as plt
import numpy as np

from posterior_flows.samplers import rwm


def main():
    def snd_func(x):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)

    def log_target(theta):
        return -0.5 * np.sum(theta**2)

    samples, accept_rate = rwm(
        log_target, theta0=1, n_steps=20_000, step_size=2, rng=np.random.default_rng(0)
    )
    x_vals = np.linspace(-4, 4, 1000)

    plt.figure()
    plt.title(rf"log_target (accept: {accept_rate:.2f})")
    plt.hist(samples, bins=50, density=True)
    plt.plot(x_vals, snd_func(x_vals))
    plt.xlabel(r"$\theta$")
    plt.show()


if __name__ == "__main__":
    main()
