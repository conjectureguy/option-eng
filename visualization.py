"""Reusable Matplotlib visualizations for option analytics."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pricing.black_scholes import OptionType, black_scholes_price
from pricing.implied_volatility import implied_volatility
from risk.greeks import black_scholes_greeks


def plot_option_price_vs_spot(
    spots: np.ndarray,
    strike: float,
    volatility: float,
    risk_free_rate: float,
    expiry: float,
    dividend_yield: float = 0.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    ax = ax or plt.subplots()[1]
    for option_type in (OptionType.CALL, OptionType.PUT):
        prices = black_scholes_price(
            option_type, spots, strike, volatility, risk_free_rate, expiry, dividend_yield
        )
        ax.plot(spots, prices, label=option_type.value.title())
    ax.set(title="European Option Price vs Spot", xlabel="Spot Price", ylabel="Option Price")
    ax.legend()
    return ax


def plot_greeks_vs_spot(
    spots: np.ndarray,
    option_type: OptionType | str,
    strike: float,
    volatility: float,
    risk_free_rate: float,
    expiry: float,
    dividend_yield: float = 0.0,
    axes: Sequence[plt.Axes] | None = None,
) -> Sequence[plt.Axes]:
    if axes is None:
        _, grid = plt.subplots(2, 3, figsize=(12, 7))
        axes = grid.flat
    frame = pd.DataFrame(
        [
            black_scholes_greeks(
                option_type,
                float(spot),
                strike,
                volatility,
                risk_free_rate,
                expiry,
                dividend_yield,
            ).as_dict()
            for spot in spots
        ]
    )
    for axis, greek in zip(axes, frame.columns):
        axis.plot(spots, frame[greek])
        axis.set(title=greek.title(), xlabel="Spot Price", ylabel=greek.title())
    return axes


def plot_monte_carlo_convergence(
    checkpoints: np.ndarray,
    estimates: np.ndarray,
    benchmark_price: float | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    ax = ax or plt.subplots()[1]
    ax.plot(checkpoints, estimates, label="Monte Carlo")
    if benchmark_price is not None:
        ax.axhline(benchmark_price, linestyle="--", color="black", label="Black-Scholes")
    ax.set(title="Monte Carlo Convergence", xlabel="Simulations", ylabel="Option Price")
    ax.legend()
    return ax


def plot_volatility_smile(
    strikes: Sequence[float],
    market_prices: Sequence[float],
    option_type: OptionType | str,
    spot: float,
    risk_free_rate: float,
    expiry: float,
    dividend_yield: float = 0.0,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    implied_volatilities = [
        implied_volatility(
            option_type,
            market_price,
            spot,
            strike,
            risk_free_rate,
            expiry,
            dividend_yield,
        )
        for strike, market_price in zip(strikes, market_prices)
    ]
    ax = ax or plt.subplots()[1]
    ax.plot(strikes, implied_volatilities, marker="o")
    ax.set(title="Volatility Smile", xlabel="Strike", ylabel="Implied Volatility")
    return ax


def plot_portfolio_pnl(
    scenarios: pd.DataFrame, ax: plt.Axes | None = None
) -> plt.Axes:
    """Plot portfolio PnL as a heat map over spot and volatility shocks."""
    table = scenarios.pivot(
        index="volatility_change", columns="spot_change", values="pnl"
    )
    ax = ax or plt.subplots()[1]
    image = ax.imshow(table.values, aspect="auto", origin="lower", cmap="coolwarm")
    ax.set(
        title="Portfolio PnL Scenarios",
        xlabel="Relative Spot Change",
        ylabel="Absolute Volatility Change",
        xticks=np.arange(len(table.columns)),
        yticks=np.arange(len(table.index)),
    )
    ax.set_xticklabels([f"{value:+.0%}" for value in table.columns])
    ax.set_yticklabels([f"{value:+.0%}" for value in table.index])
    ax.figure.colorbar(image, ax=ax, label="PnL")
    return ax

