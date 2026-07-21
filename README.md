# All Weather Portfolio & Risk Parity

A from-scratch implementation of risk-parity portfolio construction,
benchmarked against Ray Dalio's classic All Weather static allocation and
a standard 60/40 portfolio.

The core question this project explores: **allocating capital equally
across assets is not the same as allocating risk equally** — and the gap
between the two turns out to matter a lot.

## Why this exists

Built as a public companion piece to my quantitative finance work,
exploring portfolio construction methodology using synthetic
factor-driven data (no proprietary models, vendor data, or backtests are
used — everything here is original code and synthetic data).

## The four portfolios

| Portfolio | Logic |
|---|---|
| **All Weather (static)** | Ray Dalio's classic fixed weights: 30% equities, 40% long bonds, 15% intermediate bonds, 7.5% gold, 7.5% commodities |
| **Inverse volatility (rolling)** | Naive risk parity — weight inversely proportional to each asset's own volatility, ignoring correlation |
| **True risk parity (rolling)** | Equal *risk contribution* weighting — solved via the fixed-point iteration of Maillard, Roncalli & Teiletche (2010), accounting for the full covariance structure |
| **60/40 benchmark** | 60% equities / 40% long bonds, the standard institutional default |

The distinction between inverse-vol and true risk parity is the whole
point of the project: inverse-vol ignores how assets move *together*.
True risk parity accounts for it — an asset that's volatile but
diversifying (like long bonds against equities) can carry more capital
weight than its standalone volatility would suggest, because its
*contribution to total portfolio risk* is lower once correlation is
priced in.

## The data

Synthetic monthly returns for 5 asset classes (`src/simulate_assets.py`),
generated from a two-factor macro model (growth surprise, inflation
surprise) with asset-class factor loadings that encode the actual
economic logic behind the All Weather portfolio — e.g. long bonds have
negative loading on both factors (they rally when growth *and* inflation
disappoint), which is why they diversify equities. This isn't fit to any
specific historical period; it's built directly from each asset class's
textbook macro sensitivity.

## Method

1. **`src/simulate_assets.py`** — two-factor synthetic return generator
2. **`src/risk_parity.py`** — weighting schemes, all solved with numpy:
   - inverse-volatility weights (closed form)
   - true risk parity via cyclical fixed-point iteration, with numerical
     safeguards for assets whose marginal risk contribution goes negative
     (diversifiers can do this — the iteration floors and dampens rather
     than blowing up)
3. **`src/backtest.py`** — fixed-weight and rolling-reestimation backtest
   engines, plus performance metrics (CAGR, annualized vol, Sharpe,
   Sortino, max drawdown, Calmar) all implemented from scratch
4. **`run_analysis.py`** — orchestrates the full comparison and saves
   plots + a performance summary table

## Results

See `outputs/performance_summary.md` for full numbers. On this synthetic
30-year backtest:

- The **60/40 benchmark** has the highest raw CAGR but also by far the
  worst max drawdown (-23.8%) and the lowest Sharpe/Calmar — it's
  concentrated in growth risk and pays for it in the tails.
- **Inverse-vol** posts the best Calmar ratio in this run — a cheap,
  correlation-blind heuristic that still meaningfully improves
  risk-adjusted return over 60/40.
- **True risk parity** and the **static All Weather** allocation land
  between the two, trading some raw return for materially smaller
  drawdowns — which is the entire design intent of both approaches.

The weight-vs-risk-contribution chart (`outputs/weight_vs_risk_contribution.png`)
is the clearest illustration: under the static All Weather weights,
capital allocation and risk allocation are visibly *not* the same shape.
Under true risk parity, they're much closer to equal by construction.

## Repo structure

```
all-weather-portfolio/
├── src/
│   ├── simulate_assets.py    # two-factor synthetic return generator
│   ├── risk_parity.py        # weighting schemes (numpy, no scipy)
│   └── backtest.py           # backtest engines + performance metrics
├── run_analysis.py           # end-to-end pipeline entry point
├── data/                     # generated returns (csv)
├── outputs/                  # plots + performance summary
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
python run_analysis.py
```

## Extending with real data

Swap `src/simulate_assets.py` for a loader pulling real monthly total
returns (e.g. via `yfinance` for SPY/TLT/IEF/GLD/DBC as liquid proxies)
into the same `date x asset` CSV shape — everything downstream (risk
parity solver, backtest engine, metrics) is data-source agnostic.
