"""Analytic Black-Scholes Greeks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from pricing.black_scholes import OptionType, d1_d2


@dataclass(frozen=True)
class Greeks:
    """Standard option sensitivities using annualized units."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def scaled(self, quantity: float) -> "Greeks":
        return Greeks(*(quantity * value for value in self.as_tuple()))

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return self.delta, self.gamma, self.vega, self.theta, self.rho

    def as_dict(self) -> dict[str, float]:
        return dict(zip(("delta", "gamma", "vega", "theta", "rho"), self.as_tuple()))


def black_scholes_greeks(
    option_type: OptionType | str,
    spot: float,
    strike: float,
    volatility: float,
    risk_free_rate: float,
    expiry: float,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Compute analytic Delta, Gamma, Vega, Theta, and Rho."""
    kind = OptionType(option_type)
    if expiry <= 0.0 or volatility <= 0.0:
        raise ValueError("Greeks require positive volatility and expiry")
    d1, d2 = d1_d2(
        spot, strike, volatility, risk_free_rate, expiry, dividend_yield
    )
    d1_value, d2_value = float(d1), float(d2)
    discount_rate = np.exp(-risk_free_rate * expiry)
    discount_dividend = np.exp(-dividend_yield * expiry)
    density = norm.pdf(d1_value)
    sqrt_expiry = np.sqrt(expiry)

    gamma = discount_dividend * density / (spot * volatility * sqrt_expiry)
    vega = spot * discount_dividend * density * sqrt_expiry
    diffusion = -spot * discount_dividend * density * volatility / (2.0 * sqrt_expiry)
    if kind is OptionType.CALL:
        delta = discount_dividend * norm.cdf(d1_value)
        theta = (
            diffusion
            - risk_free_rate * strike * discount_rate * norm.cdf(d2_value)
            + dividend_yield * spot * discount_dividend * norm.cdf(d1_value)
        )
        rho = strike * expiry * discount_rate * norm.cdf(d2_value)
    else:
        delta = discount_dividend * (norm.cdf(d1_value) - 1.0)
        theta = (
            diffusion
            + risk_free_rate * strike * discount_rate * norm.cdf(-d2_value)
            - dividend_yield * spot * discount_dividend * norm.cdf(-d1_value)
        )
        rho = -strike * expiry * discount_rate * norm.cdf(-d2_value)
    return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))

