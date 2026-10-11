import numpy as np

from posterior_flows.models.lorenz63 import lorenz63


def test_fixed_points_are_stationary():
    sigma, rho, b = 10.0, 28.0, 8 / 3
    c = np.sqrt(b * (rho - 1))

    for p in ([0, 0, 0], [c, c, rho - 1], [-c, -c, rho - 1]):
        dx = lorenz63(0.0, np.array(p, dtype=float), sigma, rho, b)
        np.testing.assert_allclose(dx, 0.0, atol=1e-12)
