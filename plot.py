import matplotlib.pyplot as plt


def plot_attractor(t_vals, state, rho_val):
    """Plots the solution of the Lorenz-63 system"""
    x, y, z = state

    fig = plt.figure(figsize=(12, 5))

    # attractor in 3D
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(x, y, z, lw=0.5)
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.set_title(rf"Lorenz-63, $\rho$ = {rho_val}")

    # time series
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t_vals, x, label="x")
    ax2.plot(t_vals, y, label="y")
    ax2.plot(t_vals, z, label="z")
    ax2.set_xlabel("t")
    ax2.legend()

    plt.tight_layout()
    plt.show()