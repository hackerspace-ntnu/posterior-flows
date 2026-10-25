from dataclasses import dataclass


@dataclass(frozen=True)
class Lorenz63Config:
    sigma: float = 10.0
    rho_true: float = 28.0
    b: float = 8 / 3
    dt: float = 0.01
    t_burn: float = 5.0
    t_max: float = 2.0
    obs_every: int = 10
    noise_std: float = 1.0
    rho_start: float = 20.0
    rho_end: float = 35.0
    seed: int = 0
