"""Monte Carlo pricing under risk-neutral geometric Brownian motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .black_scholes import OptionType


@dataclass(frozen=True)
class MonteCarloResult:
    """Monte Carlo option-price estimate and confidence interval."""

    price: float
    standard_error: float
    confidence_interval: tuple[float, float]
    simulations: int


class MonteCarloSimulation:
    """Simulate GBM stock paths and price discounted European payoffs."""

    def __init__(
        self,
        simulations: int = 100_000,
        timesteps: int = 252,
        seed: int | None = 42,
        confidence_level: float = 0.95,
    ) -> None:
        if simulations <= 1:
            raise ValueError("simulations must exceed one")
        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if not 0.0 < confidence_level < 1.0:
            raise ValueError("confidence_level must lie between zero and one")
        self.simulations = simulations
        self.timesteps = timesteps
        self.seed = seed
        self.confidence_level = confidence_level

    def simulate_paths(
        self,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        expiry: float,
        dividend_yield: float = 0.0,
    ) -> np.ndarray:
        """Return simulated paths with shape (simulations, timesteps + 1)."""
        if spot <= 0.0 or volatility < 0.0 or expiry <= 0.0:
            raise ValueError("spot and expiry must be positive; volatility cannot be negative")
        rng = np.random.default_rng(self.seed)
        dt = expiry / self.timesteps
        draws = rng.standard_normal((self.simulations, self.timesteps))
        log_returns = (
            (risk_free_rate - dividend_yield - 0.5 * volatility**2) * dt
            + volatility * np.sqrt(dt) * draws
        )
        paths = np.empty((self.simulations, self.timesteps + 1))
        paths[:, 0] = spot
        paths[:, 1:] = spot * np.exp(np.cumsum(log_returns, axis=1))
        return paths

    def discounted_payoffs(
        self,
        option_type: OptionType | str,
        spot: float,
        strike: float,
        volatility: float,
        risk_free_rate: float,
        expiry: float,
        dividend_yield: float = 0.0,
    ) -> np.ndarray:
        """Simulate and return discounted terminal option payoffs."""
        kind = OptionType(option_type)
        if strike <= 0.0:
            raise ValueError("strike must be positive")
        terminal = self.simulate_paths(
            spot, volatility, risk_free_rate, expiry, dividend_yield
        )[:, -1]
        if kind is OptionType.CALL:
            payoffs = np.maximum(terminal - strike, 0.0)
        else:
            payoffs = np.maximum(strike - terminal, 0.0)
        return np.exp(-risk_free_rate * expiry) * payoffs

    def price(
        self,
        option_type: OptionType | str,
        spot: float,
        strike: float,
        volatility: float,
        risk_free_rate: float,
        expiry: float,
        dividend_yield: float = 0.0,
    ) -> MonteCarloResult:
        """Estimate an option price and its two-sided confidence interval."""
        discounted_payoffs = self.discounted_payoffs(
            option_type,
            spot,
            strike,
            volatility,
            risk_free_rate,
            expiry,
            dividend_yield,
        )
        estimate = float(discounted_payoffs.mean())
        standard_error = float(discounted_payoffs.std(ddof=1) / np.sqrt(self.simulations))
        critical_value = norm.ppf(0.5 + self.confidence_level / 2.0)
        margin = float(critical_value * standard_error)
        return MonteCarloResult(
            estimate,
            standard_error,
            (estimate - margin, estimate + margin),
            self.simulations,
        )

    @staticmethod
    def convergence(
        discounted_payoffs: np.ndarray, checkpoints: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return cumulative estimates at selected simulation counts."""
        payoffs = np.asarray(discounted_payoffs, dtype=float)
        if payoffs.ndim != 1 or payoffs.size == 0:
            raise ValueError("discounted_payoffs must be a non-empty one-dimensional array")
        if checkpoints is None:
            start = min(100, payoffs.size)
            checkpoints = np.unique(
                np.geomspace(start, payoffs.size, num=min(50, payoffs.size), dtype=int)
            )
        else:
            checkpoints = np.asarray(checkpoints, dtype=int)
        if np.any(checkpoints <= 0) or np.any(checkpoints > payoffs.size):
            raise ValueError("checkpoints must lie within the payoff sample")
        cumulative = np.cumsum(payoffs)
        return checkpoints, cumulative[checkpoints - 1] / checkpoints

