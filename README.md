# posterior-flows

**Work in progress.** The code, API and layout will change as the project moves
through the phases below.

Bayesian parameter estimation in chaotic dynamical systems. The long-term question
is what, for instance, viscosity and hidden initial state a 2D flow has, given a few noisy
measurements, and how this can be determined with different inference methods.
Before touching a fluid solver, the tools for the inference is built and tested on the Lorenz-63 system.

## Current status

Currently I'm implementing an ensemble Kalman filter (EnKF) for Lorenz-63, following
the literature on combined state and parameter estimation, where the unknown
parameters are appended to the state vector and updated together with it
(Evensen 2009; Evensen, Vossepoel & van Leeuwen 2022).

The end goal is to compare the methods on the same problem: accuracy and calibration
of the posterior against the number of simulations and compute time.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.14).

```bash
uv sync
uv run pre-commit install
```

## Run

```bash
uv run python experiments/phase1_lorenz/posterior_window.py
uv run pytest
```

`posterior_window.py` burns in an initial state onto the attractor, plots the
Lorenz-63 solution, simulates noisy observations, and plots the normalised log-posterior
of $\rho$ over a grid.

## Layout

```
src/posterior_flows/
  rk4.py            RK4 step and solver (single trajectory or ensemble)
  models/lorenz63.py
  observations.py   noisy observations of a true trajectory
  inference.py      Gaussian log-likelihood
  plotting.py
experiments/phase1_lorenz/   phase 1 experiments
tests/                       pytest tests
```

Part of [Hackerspace NTNU](https://github.com/hackerspace-ntnu).
