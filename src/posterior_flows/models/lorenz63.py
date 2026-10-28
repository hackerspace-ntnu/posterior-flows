import jax
import jax.numpy as jnp


def lorenz63(
    t: float, state: jax.Array, sigma: float, rho: float | jax.Array, b: float
) -> jax.Array:
    """Lorenz-63 differential system."""
    x, y, z = state
    return jnp.array([sigma * (y - x), x * (rho - z) - y, x * y - b * z])
