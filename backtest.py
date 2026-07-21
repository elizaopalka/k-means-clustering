"""
backtest.py

Rolling-rebalance portfolio backtest and standard performance metrics,
all implemented from scratch with numpy/pandas.
"""

import numpy as np
import pandas as pd


def backtest_fixed_weights(returns: pd.DataFrame, weights: np.ndarray, rebalance_freq: int = 1) -> pd.Series:
    """
    Backtest a portfolio rebalanced back to fixed target `weights` every
    `rebalance_freq` months (1 = rebalance every month). Returns the
    portfolio's monthly return series.
    """
    assets = returns.columns
    port_returns = []
    current_weights = weights.copy()

    for i, (date, row) in enumerate(returns.iterrows()):
        r = row.values
        port_ret = current_weights @ r
        port_returns.append(port_ret)

        # drift weights with realized returns, then rebalance on schedule
        drifted = current_weights * (1 + r)
        drifted = drifted / drifted.sum()

        if (i + 1) % rebalance_freq == 0:
            current_weights = weights.copy()
        else:
            current_weights = drifted

    return pd.Series(port_returns, index=returns.index, name="portfolio_return")


def backtest_rolling_risk_parity(returns: pd.DataFrame, lookback: int = 36, rebalance_freq: int = 3,
                                   weight_fn=None) -> pd.Series:
    """
    More realistic version: re-estimate the covariance matrix on a
    trailing `lookback`-month window and recompute weights every
    `rebalance_freq` months, rather than using a single fixed target.
    `weight_fn` should be a function(cov) -> weights, e.g.
    risk_parity.true_risk_parity.
    """
    port_returns = []
    current_weights = None

    for i in range(lookback, len(returns)):
        if (i - lookback) % rebalance_freq == 0 or current_weights is None:
            window = returns.iloc[i - lookback:i]
            cov = window.cov().values * 12  # annualize
            current_weights = weight_fn(cov)

        r = returns.iloc[i].values
        port_returns.append(current_weights @ r)

    idx = returns.index[lookback:]
    return pd.Series(port_returns, index=idx, name="portfolio_return")


def performance_summary(port_returns: pd.Series, rf_annual: float = 0.02) -> dict:
    monthly_rf = rf_annual / 12
    n_months = len(port_returns)
    years = n_months / 12

    cum_return = (1 + port_returns).prod()
    cagr = cum_return ** (1 / years) - 1

    ann_vol = port_returns.std() * np.sqrt(12)

    excess = port_returns - monthly_rf
    sharpe = (excess.mean() / port_returns.std()) * np.sqrt(12) if port_returns.std() > 0 else np.nan

    downside = port_returns[port_returns < monthly_rf]
    downside_dev = downside.std() * np.sqrt(12) if len(downside) > 1 else np.nan
    sortino = ((port_returns.mean() - monthly_rf) * 12) / downside_dev if downside_dev else np.nan

    wealth = (1 + port_returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth - running_max) / running_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "CAGR": round(float(cagr), 4),
        "Ann. Volatility": round(float(ann_vol), 4),
        "Sharpe Ratio": round(float(sharpe), 3),
        "Sortino Ratio": round(float(sortino), 3),
        "Max Drawdown": round(float(max_dd), 4),
        "Calmar Ratio": round(float(calmar), 3),
    }
