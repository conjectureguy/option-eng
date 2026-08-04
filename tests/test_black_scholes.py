import numpy as np
import pytest

from pricing.black_scholes import OptionType, black_scholes_price
from risk.greeks import black_scholes_greeks


def test_reference_prices() -> None:
    assert black_scholes_price("call", 100.0, 100.0, 0.2, 0.05, 1.0) == pytest.approx(
        10.4506, abs=1e-4
    )
    assert black_scholes_price("put", 100.0, 100.0, 0.2, 0.05, 1.0) == pytest.approx(
        5.5735, abs=1e-4
    )


def test_put_call_parity_with_dividend_yield() -> None:
    spot, strike, rate, dividend, expiry = 105.0, 100.0, 0.04, 0.015, 1.25
    call = black_scholes_price("call", spot, strike, 0.3, rate, expiry, dividend)
    put = black_scholes_price("put", spot, strike, 0.3, rate, expiry, dividend)
    expected = spot * np.exp(-dividend * expiry) - strike * np.exp(-rate * expiry)
    assert call - put == pytest.approx(expected)


def test_reference_call_greeks() -> None:
    greeks = black_scholes_greeks(OptionType.CALL, 100.0, 100.0, 0.2, 0.05, 1.0)
    assert greeks.delta == pytest.approx(0.636831, abs=1e-6)
    assert greeks.gamma == pytest.approx(0.018762, abs=1e-6)
    assert greeks.vega == pytest.approx(37.5240, abs=1e-4)
    assert greeks.theta == pytest.approx(-6.41403, abs=1e-5)
    assert greeks.rho == pytest.approx(53.2325, abs=1e-4)


def test_vectorized_spot_prices_and_expiry_payoff() -> None:
    prices = black_scholes_price("call", np.array([90.0, 100.0, 110.0]), 100.0, 0.2, 0.05, 1.0)
    assert prices.shape == (3,)
    assert black_scholes_price("put", 90.0, 100.0, 0.2, 0.05, 0.0) == 10.0

