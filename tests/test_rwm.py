import numpy as np

from posterior_flows.samplers import rwm


def test_samples_initial():
    theta0_1 = 1.5
    theta0_2 = [1.0, 2.0, 3.0]
    n_steps = 10

    rng = np.random.default_rng(0)

    samples_1, _ = rwm(
        log_target=lambda x: 0.0, theta0=theta0_1, n_steps=n_steps, step_size=1, rng=rng
    )
    samples_2, _ = rwm(
        log_target=lambda x: 0.0, theta0=theta0_2, n_steps=n_steps, step_size=1, rng=rng
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

    rng = np.random.default_rng(0)

    def log_target(x):
        # only the starting point has finite density
        return 0.0 if np.array_equal(x, theta0) else -np.inf

    samples, accept_rate = rwm(
        log_target=log_target, theta0=theta0, n_steps=n_steps, step_size=1, rng=rng
    )

    # every proposal is rejected
    assert accept_rate == 0.0

    # rejected steps repeat the current point
    np.testing.assert_array_equal(samples, np.tile(theta0, (n_steps, 1)))


def test_flat_target_accepts_all():
    step_size = 0.7
    n_steps = 5000

    rng = np.random.default_rng(0)

    samples, accept_rate = rwm(
        log_target=lambda x: 0.0,
        theta0=0.0,
        n_steps=n_steps,
        step_size=step_size,
        rng=rng,
    )

    # every proposal is equally likely, so all are accepted
    assert accept_rate == 1.0

    # pure random walk: increments have std equal to step_size
    increments = np.diff(samples[:, 0])
    np.testing.assert_allclose(increments.std(), step_size, rtol=0.05)


def test_one_log_target_call_per_step():
    n_steps = 100
    calls = [0]  # list, so the inner function can modify it

    def log_target(x):
        calls[0] += 1
        return -0.5 * np.sum(x**2)

    rng = np.random.default_rng(0)
    rwm(log_target=log_target, theta0=0.0, n_steps=n_steps, step_size=1, rng=rng)

    # one call for the start, then one per proposal
    assert calls[0] == n_steps


def test_same_seed_same_chain():
    def log_target(x):
        return -0.5 * np.sum(x**2)

    samples_1, acc_1 = rwm(
        log_target,
        theta0=1.0,
        n_steps=200,
        step_size=1,
        rng=np.random.default_rng(42),
    )
    samples_2, acc_2 = rwm(
        log_target,
        theta0=1.0,
        n_steps=200,
        step_size=1,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(samples_1, samples_2)
    assert acc_1 == acc_2


def test_gaussian_moments():
    mu, sigma = 3.0, 2.0
    n_steps = 20_000
    burn_in = 1_000

    def log_target(x):
        return -0.5 * np.sum(((x - mu) / sigma) ** 2)

    rng = np.random.default_rng(0)
    samples, accept_rate = rwm(
        log_target, theta0=0.0, n_steps=n_steps, step_size=2.5 * sigma, rng=rng
    )
    x = samples[burn_in:, 0]

    # step size close to optimal for 1D
    assert 0.3 < accept_rate < 0.6

    # tolerances are ~5 standard errors, measured over 200 seeds
    # (the chain's standard error is ~2x that of independent draws)
    assert abs(x.mean() - mu) < 0.15
    assert abs(x.var() - sigma**2) < 0.45


def test_2d_gaussian_different_scales():
    mu = np.array([0.0, 5.0])
    sigma = np.array([1.0, 2.0])
    n_steps = 30_000
    burn_in = 1_000

    def log_target(x):
        return -0.5 * np.sum(((x - mu) / sigma) ** 2)

    rng = np.random.default_rng(0)
    samples, _ = rwm(
        log_target, theta0=[0.0, 0.0], n_steps=n_steps, step_size=1.5, rng=rng
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
        return -np.inf if x[0] < 0 else -x[0]

    rng = np.random.default_rng(0)
    samples, _ = rwm(log_target, theta0=1.0, n_steps=n_steps, step_size=1.5, rng=rng)
    x = samples[burn_in:, 0]

    # no sample outside the support
    assert np.all(x >= 0)

    # exponential(1) has mean 1
    assert abs(x.mean() - 1.0) < 0.1
