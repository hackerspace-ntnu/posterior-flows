# posterior-flows

Work in progress.

Bayesian parameter estimation in chaotic dynamical systems. Phase 1 estimates
$\rho$ in the Lorenz-63 system from noisy observation. The long-term goal is
inference for 2D fluid flows.

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

## Layout

- `src/posterior_flows/`: RK4 integrator, Lorenz-63 model, observation simulation, plotting
- `experiments/phase1_lorenz/`: phase 1 experiments
- `tests/`: tests

Part of [Hackerspace NTNU](https://github.com/hackerspace-ntnu).
