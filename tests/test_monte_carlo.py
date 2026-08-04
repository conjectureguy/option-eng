import numpy as np

from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import MonteCarloSimulation


def test_simulates_configured_path_shape() -> None:
    simulation = MonteCarloSimulation(simulations=500, timesteps=12, seed=7)
    paths = simulation.simulate_paths(100.0, 0.2, 0.05, 1.0)
    assert paths.shape == (500, 13)
    assert np.all(paths[:, 0] == 100.0)


def test_price_confidence_interval_contains_black_scholes_reference() -> None:
    simulation = MonteCarloSimulation(simulations=80_000, timesteps=12, seed=7)
    result = simulation.price("call", 100.0, 100.0, 0.2, 0.05, 1.0)
    benchmark = black_scholes_price("call", 100.0, 100.0, 0.2, 0.05, 1.0)
    assert result.confidence_interval[0] < benchmark < result.confidence_interval[1]
    assert result.standard_error > 0.0


def test_convergence_returns_requested_checkpoints() -> None:
    checkpoints, estimates = MonteCarloSimulation.convergence(
        np.array([1.0, 2.0, 3.0, 4.0]), np.array([1, 2, 4])
    )
    assert checkpoints.tolist() == [1, 2, 4]
    assert estimates.tolist() == [1.0, 1.5, 2.5]

