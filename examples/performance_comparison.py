"""Compare closed-form and Monte Carlo pricing latency and accuracy."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import MonteCarloSimulation


def main() -> None:
    inputs = ("call", 100.0, 100.0, 0.2, 0.05, 1.0)

    started = perf_counter()
    black_scholes = black_scholes_price(*inputs)
    analytic_seconds = perf_counter() - started

    simulation = MonteCarloSimulation(simulations=100_000, timesteps=252, seed=42)
    started = perf_counter()
    monte_carlo = simulation.price(*inputs)
    monte_carlo_seconds = perf_counter() - started

    print(f"Black-Scholes: {black_scholes:.6f} in {analytic_seconds:.6f}s")
    print(
        f"Monte Carlo:   {monte_carlo.price:.6f} in {monte_carlo_seconds:.6f}s "
        f"(95% CI: {monte_carlo.confidence_interval})"
    )
    print(f"Absolute error: {abs(monte_carlo.price - black_scholes):.6f}")


if __name__ == "__main__":
    main()
