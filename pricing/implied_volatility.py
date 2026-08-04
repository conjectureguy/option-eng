"""Implied-volatility solvers for European options."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.optimize import bisect

from risk.greeks import black_scholes_greeks

from .black_scholes import OptionType, black_scholes_price


def _validate_market_price(
    option_type: OptionType,
    market_price: float,
    spot: float,
    strike: float,
    risk_free_rate: float,
    expiry: float,
    dividend_yield: float,
) -> None:
    if expiry <= 0.0:
        raise ValueError("expiry must be positive")
    discounted_spot = spot * np.exp(-dividend_yield * expiry)
    discounted_strike = strike * np.exp(-risk_free_rate * expiry)
    lower = (
        max(discounted_spot - discounted_strike, 0.0)
        if option_type is OptionType.CALL
        else max(discounted_strike - discounted_spot, 0.0)
    )
    upper = discounted_spot if option_type is OptionType.CALL else discounted_strike
    if not lower <= market_price <= upper:
        raise ValueError("market_price violates no-arbitrage bounds")


def implied_volatility(
    option_type: OptionType | str,
    market_price: float,
    spot: float,
    strike: float,
    risk_free_rate: float,
    expiry: float,
    dividend_yield: float = 0.0,
    *,
    method: Literal["newton", "bisection"] = "newton",
    initial_guess: float = 0.2,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    min_volatility: float = 1e-8,
    max_volatility: float = 5.0,
) -> float:
    """Recover volatility from an observed option price."""
    kind = OptionType(option_type)
    _validate_market_price(
        kind, market_price, spot, strike, risk_free_rate, expiry, dividend_yield
    )
    if tolerance <= 0.0 or max_iterations <= 0 or min_volatility >= max_volatility:
        raise ValueError("invalid solver configuration")

    def price_error(volatility: float) -> float:
        return (
            black_scholes_price(
                kind,
                spot,
                strike,
                volatility,
                risk_free_rate,
                expiry,
                dividend_yield,
            )
            - market_price
        )

    if method == "newton":
        volatility = float(np.clip(initial_guess, min_volatility, max_volatility))
        for _ in range(max_iterations):
            error = price_error(volatility)
            if abs(error) <= tolerance:
                return volatility
            vega = black_scholes_greeks(
                kind,
                spot,
                strike,
                volatility,
                risk_free_rate,
                expiry,
                dividend_yield,
            ).vega
            if vega <= 1e-12:
                break
            next_volatility = volatility - error / vega
            if not min_volatility < next_volatility < max_volatility:
                break
            volatility = next_volatility
    elif method != "bisection":
        raise ValueError("method must be 'newton' or 'bisection'")

    if price_error(min_volatility) >= 0.0:
        return min_volatility
    if price_error(max_volatility) < 0.0:
        raise ValueError("implied volatility exceeds max_volatility")
    return float(
        bisect(
            price_error,
            min_volatility,
            max_volatility,
            xtol=tolerance,
            maxiter=max_iterations,
        )
    )

