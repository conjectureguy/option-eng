import pytest

from pricing.black_scholes import black_scholes_price
from pricing.implied_volatility import implied_volatility


@pytest.mark.parametrize("method", ["newton", "bisection"])
def test_recovers_known_volatility(method: str) -> None:
    expected = 0.32
    price = black_scholes_price("put", 100.0, 105.0, expected, 0.03, 1.5, 0.01)
    actual = implied_volatility(
        "put", price, 100.0, 105.0, 0.03, 1.5, 0.01, method=method
    )
    assert actual == pytest.approx(expected, abs=1e-7)


def test_rejects_price_outside_arbitrage_bounds() -> None:
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility("call", 101.0, 100.0, 100.0, 0.05, 1.0)

