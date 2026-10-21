import jax.numpy as jnp


def lorenz63(
    t: float, state: jnp.ndarray, sigma: float, rho: float | jnp.ndarray, b: float
) -> jnp.ndarray:
    """Lorenz-63 differential system."""
    x, y, z = state
    return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - b * z])
