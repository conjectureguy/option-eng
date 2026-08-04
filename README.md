# Python Options Pricing Engine

A quantitative-finance project for European option valuation, numerical methods, portfolio risk,
and visualization. The implementation favors explicit formulas and small composable APIs so that
the numerical assumptions remain easy to inspect.

## Features

- Black-Scholes call and put pricing, including continuous dividend yield
- Analytic Delta, Gamma, Vega, Theta, and Rho
- Vectorized option-price evaluation across spot values
- Geometric Brownian Motion path simulation with configurable paths, timesteps, and random seed
- Monte Carlo price estimates, standard errors, confidence intervals, and convergence analysis
- Newton-Raphson implied volatility with bisection support and fallback
- Portfolio valuation, signed position aggregation, and spot-volatility scenario analysis
- Reusable Matplotlib plots and an example research notebook
- pytest coverage for analytic formulas, solvers, simulation behavior, and portfolios

## Layout

```text
eng-py/
├── pricing/
│   ├── black_scholes.py
│   ├── monte_carlo.py
│   └── implied_volatility.py
├── risk/
│   ├── greeks.py
│   └── portfolio.py
├── notebooks/
│   └── analysis.ipynb
├── examples/
│   └── performance_comparison.py
├── tests/
├── visualization.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Mathematics

For spot \(S\), strike \(K\), time to expiry \(T\), volatility \(\sigma\), continuously compounded
risk-free rate \(r\), and dividend yield \(q\):

\[
d_1 = \frac{\ln(S/K) + (r-q+\sigma^2/2)T}{\sigma\sqrt{T}}, \qquad
d_2 = d_1 - \sigma\sqrt{T}
\]

European call and put prices are:

\[
C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2)
\]

\[
P = K e^{-rT}N(-d_2) - S e^{-qT}N(-d_1)
\]

The Greek formulas are analytic derivatives of these prices. Vega and Rho use unit changes in
volatility and interest rate; divide them by 100 to report sensitivity to a one percentage-point
move. Theta is annualized calendar-time decay.

Monte Carlo simulation uses the risk-neutral Geometric Brownian Motion process:

\[
S_{t+\Delta t} =
S_t \exp((r-q-\sigma^2/2)\Delta t + \sigma\sqrt{\Delta t}Z), \qquad Z \sim N(0, 1)
\]

The discounted terminal payoff mean estimates the option price. The engine also reports the
sample standard error and a configurable two-sided confidence interval.

Implied volatility numerically inverts the Black-Scholes price. Newton-Raphson uses analytic Vega
and falls back to bisection if its update becomes unstable. Both methods enforce no-arbitrage price
bounds before solving.

## Setup

```bash
cd eng-py
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the test suite:

```bash
python -m pytest -q
```

Run the performance comparison:

```bash
python examples/performance_comparison.py
```

## Example

```python
from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import MonteCarloSimulation
from risk.portfolio import OptionPosition, Portfolio

price = black_scholes_price("call", 100.0, 100.0, 0.20, 0.05, 1.0)
simulation = MonteCarloSimulation(simulations=100_000, timesteps=252, seed=42)
estimate = simulation.price("call", 100.0, 100.0, 0.20, 0.05, 1.0)

portfolio = Portfolio([
    OptionPosition("call", strike=100.0, expiry=1.0, quantity=10),
    OptionPosition("put", strike=95.0, expiry=0.5, quantity=-5),
])
risk = portfolio.aggregate_greeks(spot=100.0, volatility=0.20, risk_free_rate=0.05)
```

## Research Notebook

[`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) walks through:

- Black-Scholes prices and Greeks
- Monte Carlo pricing and convergence against the closed-form benchmark
- Implied-volatility smile recovery from synthetic market prices
- Portfolio risk aggregation and scenario PnL visualization
- Runtime comparison between closed-form and simulation-based pricing

