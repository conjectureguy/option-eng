import pytest

from pricing.black_scholes import black_scholes_price
from risk.greeks import black_scholes_greeks
from risk.portfolio import OptionPosition, Portfolio


def test_aggregates_value_and_greeks() -> None:
    portfolio = Portfolio(
        [
            OptionPosition("call", 100.0, 1.0, quantity=3.0),
            OptionPosition("put", 100.0, 1.0, quantity=-2.0),
        ]
    )
    expected_value = 3.0 * black_scholes_price("call", 100.0, 100.0, 0.2, 0.05, 1.0)
    expected_value -= 2.0 * black_scholes_price("put", 100.0, 100.0, 0.2, 0.05, 1.0)
    expected_delta = 3.0 * black_scholes_greeks("call", 100.0, 100.0, 0.2, 0.05, 1.0).delta
    expected_delta -= 2.0 * black_scholes_greeks("put", 100.0, 100.0, 0.2, 0.05, 1.0).delta

    assert portfolio.value(100.0, 0.2, 0.05) == pytest.approx(expected_value)
    assert portfolio.aggregate_greeks(100.0, 0.2, 0.05).delta == pytest.approx(expected_delta)


def test_scenario_analysis_returns_pnl_grid() -> None:
    portfolio = Portfolio([OptionPosition("call", 100.0, 1.0, quantity=1.0)])
    scenarios = portfolio.scenario_analysis(
        100.0, 0.2, 0.05, spot_changes=[-0.1, 0.0, 0.1], volatility_changes=[-0.05, 0.0, 0.05]
    )
    assert len(scenarios) == 9
    baseline = scenarios.query("spot_change == 0 and volatility_change == 0").iloc[0]
    assert baseline["pnl"] == pytest.approx(0.0)

