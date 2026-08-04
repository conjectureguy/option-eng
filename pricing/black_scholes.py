"""Black-Scholes pricing for European call and put options."""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy.stats import norm


class OptionType(str, Enum):
    """Supported European option payoff types."""

    CALL = "call"
    PUT = "put"


def _validate_inputs(
    spot: float | np.ndarray,
    strike: float,
    volatility: float,
    risk_free_rate: float,
    expiry: float,
) -> np.ndarray:
    del risk_free_rate
    spots = np.asarray(spot, dtype=float)
    if np.any(spots <= 0.0):
        raise ValueError("spot must be positive")
    if strike <= 0.0:
        raise ValueError("strike must be positive")
    if volatility < 0.0:
        raise ValueError("volatility must be non-negative")
    if expiry < 0.0:
        raise ValueError("expiry must be non-negative")
    return spots


def _intrinsic_value(
    option_type: OptionType, spot: np.ndarray, strike: float
) -> np.ndarray:
    if option_type is OptionType.CALL:
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)


def d1_d2(
    spot: float | np.ndarray,
    strike: float,
    volatility: float,
    risk_free_rate: float,
    expiry: float,
    dividend_yield: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Black-Scholes d1 and d2 terms."""
    spots = _validate_inputs(spot, strike, volatility, risk_free_rate, expiry)
    if volatility == 0.0 or expiry == 0.0:
        raise ValueError("d1 and d2 are undefined for zero volatility or expiry")
    sqrt_expiry = np.sqrt(expiry)
    d1 = (
        np.log(spots / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * expiry
    ) / (volatility * sqrt_expiry)
    return d1, d1 - volatility * sqrt_expiry


def black_scholes_price(
    option_type: OptionType | str,
    spot: float | np.ndarray,
    strike: float,
    volatility: float,
    risk_free_rate: float,
    expiry: float,
    dividend_yield: float = 0.0,
) -> float | np.ndarray:
    """Price a European option using the Black-Scholes formula."""
    kind = OptionType(option_type)
    spots = _validate_inputs(spot, strike, volatility, risk_free_rate, expiry)
    if expiry == 0.0:
        result = _intrinsic_value(kind, spots, strike)
    elif volatility == 0.0:
        discounted_spot = spots * np.exp(-dividend_yield * expiry)
        discounted_strike = strike * np.exp(-risk_free_rate * expiry)
        result = _intrinsic_value(kind, discounted_spot, discounted_strike)
    else:
        d1, d2 = d1_d2(
            spots, strike, volatility, risk_free_rate, expiry, dividend_yield
        )
        discounted_spot = spots * np.exp(-dividend_yield * expiry)
        discounted_strike = strike * np.exp(-risk_free_rate * expiry)
        if kind is OptionType.CALL:
            result = discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
        else:
            result = discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)
    return float(result) if result.ndim == 0 else result

