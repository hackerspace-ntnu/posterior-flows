import numpy as np


def lorenz63(
    t: float, state: np.ndarray, sigma: float, rho: float | np.ndarray, b: float
) -> np.ndarray:
    """Lorenz-63 differential system."""
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - b * z])
