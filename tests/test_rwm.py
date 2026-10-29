import jax
import jax.numpy as jnp
import numpy as np

from posterior_flows.inference import (
    gaussian_log_likelihood,
    make_lorenz63_log_likelihood,
    make_unpack_rho_x0,
)
from posterior_flows.models.lorenz63 import lorenz63
from posterior_flows.observations import obs_simulation
from posterior_flows.rk4 import rk4_solve
from posterior_flows.samplers import rwm


def test_samples_initial():
    theta0_1 = 1.5
    theta0_2 = [1.0, 2.0, 3.0]
    n_steps = 10

    key_1, key_2 = jax.random.split(jax.random.key(0))

    samples_1, _ = rwm(
        log_target=lambda x: jnp.zeros(()),
        theta0=theta0_1,
        n_steps=n_steps,
        step_size=1,
        key=key_1,
    )
    samples_2, _ = rwm(
        log_target=lambda x: jnp.zeros(()),
        theta0=theta0_2,
        n_steps=n_steps,
        step_size=1,
        key=key_2,
    )

    # shapes are the same
    assert samples_1.shape == (n_steps, 1)
    assert samples_2.shape == (n_steps, len(theta0_2))

    # correct initial theta
    np.testing.assert_array_equal(samples_1[0], theta0_1)
    np.testing.assert_array_equal(samples_2[0], theta0_2)


def test_all_rejected_chain_stays():
    theta0 = [2.5, -1.0]  # nonzero, so a missing else-branch leaves visible zeros
    n_steps = 50

    def log_target(x):
        # only the starting point has finite density
        return jnp.where(jnp.all(x == jnp.array(theta0)), 0.0, -jnp.inf)

    samples, accept_rate = rwm(
        log_target=log_target,
        theta0=theta0,
        n_steps=n_steps,
        step_size=1,
        key=jax.random.key(0),
    )

    # every proposal is rejected
    assert accept_rate == 0.0

    # rejected steps repeat the current point
    np.testing.assert_array_equal(samples, np.tile(theta0, (n_steps, 1)))


def test_flat_target_accepts_all():
    step_size = 0.7
    n_steps = 5000

    samples, accept_rate = rwm(
        log_target=lambda x: jnp.zeros(()),
        theta0=0.0,
        n_steps=n_steps,
        step_size=step_size,
        key=jax.random.key(0),
    )

    # every proposal is equally likely, so all are accepted
    assert accept_rate == 1.0

    # pure random walk: increments have std equal to step_size
    increments = jnp.diff(samples[:, 0])
    np.testing.assert_allclose(increments.std(), step_size, rtol=0.05)


def test_same_seed_same_chain():
    def log_target(x):
        return -0.5 * jnp.sum(x**2)

    samples_1, acc_1 = rwm(
        log_target, theta0=1.0, n_steps=200, step_size=1, key=jax.random.key(42)
    )
    samples_2, acc_2 = rwm(
        log_target, theta0=1.0, n_steps=200, step_size=1, key=jax.random.key(42)
    )

    np.testing.assert_array_equal(samples_1, samples_2)
    assert acc_1 == acc_2


def test_gaussian_moments():
    mu, sigma = 3.0, 2.0
    n_steps = 20_000
    burn_in = 1_000

    def log_target(x):
        return -0.5 * jnp.sum(((x - mu) / sigma) ** 2)

    samples, accept_rate = rwm(
        log_target,
        theta0=0.0,
        n_steps=n_steps,
        step_size=2.5 * sigma,
        key=jax.random.key(0),
    )
    x = samples[burn_in:, 0]

    # step size close to optimal for 1D
    assert 0.3 < accept_rate < 0.6

    # tolerances are approx 5 standard errors
    # (the chain's standard error is approx 2x that of independent draws)
    assert abs(x.mean() - mu) < 0.15
    assert abs(x.var() - sigma**2) < 0.45


def test_2d_gaussian_different_scales():
    mu = jnp.array([0.0, 5.0])
    sigma = jnp.array([1.0, 2.0])
    n_steps = 30_000
    burn_in = 1_000

    def log_target(x):
        return -0.5 * jnp.sum(((x - mu) / sigma) ** 2)

    samples, _ = rwm(
        log_target,
        theta0=[0.0, 0.0],
        n_steps=n_steps,
        step_size=1.5,
        key=jax.random.key(0),
    )
    x = samples[burn_in:]

    # each component gets its own mean and variance
    np.testing.assert_allclose(x.mean(axis=0), mu, atol=0.25)
    np.testing.assert_allclose(x.var(axis=0), sigma**2, rtol=0.15)


def test_respects_support():
    n_steps = 20_000
    burn_in = 1_000

    def log_target(x):
        # exponential distribution with rate 1, zero density for x < 0
        return jnp.where(x[0] < 0, -jnp.inf, -x[0])

    samples, _ = rwm(
        log_target, theta0=1.0, n_steps=n_steps, step_size=1.5, key=jax.random.key(0)
    )
    x = samples[burn_in:, 0]

    # no sample outside the support
    assert jnp.all(x >= 0)

    # exponential(1) has mean 1
    assert abs(x.mean() - 1.0) < 0.1


# Tests for batching

SIGMA, RHO, B = 10.0, 28.0, 8 / 3
DT = 0.01
T_MAX = 0.5
OBS_EVERY = 10
NOISE_STD = 1.0

# N = 4 trajectories with d = 3, so N != d and swapped axes give shape errors
X0S = jnp.array(
    [
        [1.0, 1.0, 1.0],
        [2.0, 0.5, 10.0],
        [-3.0, 4.0, 20.0],
        [0.0, -1.0, 5.0],
    ]
)  # (N, d)


def test_rk4_batch_matches_single():
    t, Y = rk4_solve(
        lorenz63, [0, T_MAX], X0S.T, DT, args=(SIGMA, RHO, B), save_every=OBS_EVERY
    )

    # time on the middle axis, trajectories last
    assert Y.shape == (3, len(t), len(X0S))

    # each column of the batch equals its own single run
    for i, x0 in enumerate(X0S):
        _, Y_i = rk4_solve(
            lorenz63, [0, T_MAX], x0, DT, args=(SIGMA, RHO, B), save_every=OBS_EVERY
        )
        np.testing.assert_allclose(Y[..., i], Y_i, rtol=1e-10)


def test_rk4_batch_rho_matches_single():
    rhos = jnp.array([20.0, 25.0, 30.0, 35.0])
    x0 = jnp.array([1.0, 1.0, 1.0])
    x0s = jnp.tile(x0[:, None], (1, len(rhos)))  # (d, N), same start for all

    _, Y = rk4_solve(
        lorenz63, [0, T_MAX], x0s, DT, args=(SIGMA, rhos, B), save_every=OBS_EVERY
    )

    # rho (N,) must pair with trajectory i, not broadcast across the state
    for i, rho in enumerate(rhos):
        _, Y_i = rk4_solve(
            lorenz63, [0, T_MAX], x0, DT, args=(SIGMA, rho, B), save_every=OBS_EVERY
        )
        np.testing.assert_allclose(Y[..., i], Y_i, rtol=1e-10)


def test_obs_simulation_batch_shapes():
    t_obs, y_obs, x_true = obs_simulation(
        lorenz63,
        X0S.T,
        T_MAX,
        DT,
        OBS_EVERY,
        args=(SIGMA, RHO, B),
        noise_std=NOISE_STD,
        key=jax.random.key(0),
    )

    assert y_obs.shape == x_true.shape == (3, len(t_obs), len(X0S))

    # noise is drawn per trajectory, not one draw broadcast to all
    noise = y_obs - x_true
    assert not np.allclose(noise[..., 0], noise[..., 1])


def test_gaussian_log_likelihood_batch_matches_single():
    n_batch, d, n_obs = 4, 3, 11

    key_obs, key_pred = jax.random.split(jax.random.key(0))
    y_obs = jax.random.normal(key_obs, (d, n_obs))
    x_pred = jax.random.normal(key_pred, (n_batch, d, n_obs))
    noise_std = jnp.array([[0.5], [1.0], [2.0]])  # per-component, (d, 1)

    for include_constant in [False, True]:
        ll = gaussian_log_likelihood(y_obs, x_pred, noise_std, include_constant)
        assert ll.shape == (n_batch,)

        for i in range(n_batch):
            ll_i = gaussian_log_likelihood(
                y_obs, x_pred[i], noise_std, include_constant
            )

            # a single trajectory gives a scalar
            assert ll_i.shape == ()
            np.testing.assert_allclose(ll[i], ll_i, rtol=1e-12)


def test_lorenz63_log_likelihood_batch_matches_single():
    _, y_obs, _ = obs_simulation(
        lorenz63,
        jnp.array([1.0, 1.0, 1.0]),
        T_MAX,
        DT,
        OBS_EVERY,
        args=(SIGMA, RHO, B),
        noise_std=NOISE_STD,
        key=jax.random.key(0),
    )

    unpack = make_unpack_rho_x0(SIGMA, B)
    log_likelihood = make_lorenz63_log_likelihood(
        unpack, T_MAX, DT, OBS_EVERY, NOISE_STD
    )

    # theta = [rho, x0_x, x0_y, x0_z], N = 5 so N differs from both 3 and 4
    thetas = jnp.array(
        [
            [28.0, 1.0, 1.0, 1.0],
            [25.0, 1.5, 0.5, 2.0],
            [30.0, 0.0, 2.0, 1.0],
            [22.0, -1.0, 1.0, 3.0],
            [34.0, 2.0, -0.5, 0.5],
        ]
    )

    ll = log_likelihood(thetas, y_obs)
    assert ll.shape == (len(thetas),)

    for i, theta in enumerate(thetas):
        ll_i = log_likelihood(theta, y_obs)

        # a single theta gives a scalar
        assert ll_i.shape == ()
        np.testing.assert_allclose(ll[i], ll_i, rtol=1e-10)


def test_rwm_vmap_matches_single():
    n_chains, n_steps, step_size = 4, 500, 1.0

    def log_target(x):
        return -0.5 * jnp.sum(x**2)

    keys = jax.random.split(jax.random.key(0), n_chains)
    theta0s = jnp.array([[-2.0, 0.0], [0.0, 1.0], [1.0, -1.0], [3.0, 2.0]])

    run = jax.vmap(lambda theta0, key: rwm(log_target, theta0, n_steps, step_size, key))
    samples, accept_rates = run(theta0s, keys)

    assert samples.shape == (n_chains, n_steps, 2)
    assert accept_rates.shape == (n_chains,)

    # chains under vmap are the same as running them one by one
    for i in range(n_chains):
        samples_i, acc_i = rwm(log_target, theta0s[i], n_steps, step_size, keys[i])
        np.testing.assert_allclose(samples[i], samples_i, rtol=1e-12)
        np.testing.assert_allclose(accept_rates[i], acc_i)
