"""Portfolio valuation, aggregate Greeks, and scenario analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pricing.black_scholes import OptionType, black_scholes_price
from risk.greeks import Greeks, black_scholes_greeks


@dataclass(frozen=True)
class OptionPosition:
    """A European option contract and signed portfolio quantity."""

    option_type: OptionType | str
    strike: float
    expiry: float
    quantity: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        if self.strike <= 0.0:
            raise ValueError("strike must be positive")
        if self.expiry <= 0.0:
            raise ValueError("expiry must be positive")


class Portfolio:
    """Collection of European option positions valued with Black-Scholes."""

    def __init__(self, positions: list[OptionPosition] | None = None) -> None:
        self.positions = list(positions or [])

    def add(self, position: OptionPosition) -> None:
        self.positions.append(position)

    def value(
        self,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> float:
        return float(
            sum(
                position.quantity
                * black_scholes_price(
                    position.option_type,
                    spot,
                    position.strike,
                    volatility,
                    risk_free_rate,
                    position.expiry,
                    dividend_yield,
                )
                for position in self.positions
            )
        )

    def aggregate_greeks(
        self,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
    ) -> Greeks:
        totals = [0.0] * 5
        for position in self.positions:
            scaled = black_scholes_greeks(
                position.option_type,
                spot,
                position.strike,
                volatility,
                risk_free_rate,
                position.expiry,
                dividend_yield,
            ).scaled(position.quantity)
            totals = [total + value for total, value in zip(totals, scaled.as_tuple())]
        return Greeks(*totals)

    def scenario_analysis(
        self,
        spot: float,
        volatility: float,
        risk_free_rate: float,
        spot_changes: list[float],
        volatility_changes: list[float],
        dividend_yield: float = 0.0,
    ) -> pd.DataFrame:
        """Return portfolio value and PnL for relative spot and absolute volatility shocks."""
        baseline = self.value(spot, volatility, risk_free_rate, dividend_yield)
        rows = []
        for spot_change in spot_changes:
            for volatility_change in volatility_changes:
                scenario_spot = spot * (1.0 + spot_change)
                scenario_volatility = volatility + volatility_change
                if scenario_spot <= 0.0 or scenario_volatility < 0.0:
                    raise ValueError("scenario shocks produce invalid market inputs")
                scenario_value = self.value(
                    scenario_spot,
                    scenario_volatility,
                    risk_free_rate,
                    dividend_yield,
                )
                rows.append(
                    {
                        "spot_change": spot_change,
                        "volatility_change": volatility_change,
                        "spot": scenario_spot,
                        "volatility": scenario_volatility,
                        "portfolio_value": scenario_value,
                        "pnl": scenario_value - baseline,
                    }
                )
        return pd.DataFrame(rows)

