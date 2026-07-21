"""
run_analysis.py

End-to-end pipeline:
  1. Simulate (or load) synthetic monthly returns for 5 asset classes
  2. Build four portfolios: classic All Weather, naive inverse-vol,
     true risk parity (equal risk contribution), and a 60/40 benchmark
  3. Backtest each with monthly rebalancing
  4. Score CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar
  5. Save comparison plots to outputs/

Run: python run_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulate_assets import simulate
from risk_parity import inverse_vol_weights, true_risk_parity, all_weather_weights
from backtest import backtest_fixed_weights, backtest_rolling_risk_parity, performance_summary

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "synthetic_asset_returns.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "font.size": 10,
})

COLORS = {
    "All Weather (static)": "#C97D3E",
    "Risk Parity (rolling)": "#4A6FA5",
    "Inverse Vol (rolling)": "#639922",
    "60/40 Benchmark": "#888780",
}


def main():
    if DATA_PATH.exists():
        returns = pd.read_csv(DATA_PATH, index_col="date", parse_dates=True)
    else:
        returns = simulate()
        DATA_PATH.parent.mkdir(exist_ok=True)
        returns.to_csv(DATA_PATH)

    print(f"Loaded returns: {returns.shape[0]} months x {returns.shape[1]} assets")
    assets = list(returns.columns)

    # --- Portfolio 1: classic static All Weather allocation ---
    aw_weights = all_weather_weights(assets)
    aw_returns = backtest_fixed_weights(returns, aw_weights, rebalance_freq=1)

    # --- Portfolio 2: rolling true risk parity ---
    rp_returns = backtest_rolling_risk_parity(returns, lookback=36, rebalance_freq=3, weight_fn=true_risk_parity)

    # --- Portfolio 3: rolling inverse-vol (naive risk parity) ---
    ivol_returns = backtest_rolling_risk_parity(returns, lookback=36, rebalance_freq=3, weight_fn=inverse_vol_weights)

    # --- Portfolio 4: 60/40 benchmark (equities / long bonds) ---
    sixty_forty = np.zeros(len(assets))
    sixty_forty[assets.index("EQ")] = 0.60
    sixty_forty[assets.index("LTB")] = 0.40
    bench_returns = backtest_fixed_weights(returns, sixty_forty, rebalance_freq=1)

    portfolios = {
        "All Weather (static)": aw_returns,
        "Risk Parity (rolling)": rp_returns,
        "Inverse Vol (rolling)": ivol_returns,
        "60/40 Benchmark": bench_returns,
    }

    # --- Performance table ---
    summary = {}
    for name, r in portfolios.items():
        summary[name] = performance_summary(r)
    summary_df = pd.DataFrame(summary).T
    print("\n--- Performance summary ---")
    print(summary_df)
    summary_df.to_csv(OUT_DIR / "performance_summary.csv")

    with open(OUT_DIR / "performance_summary.md", "w") as f:
        f.write("# Portfolio performance summary\n\n")
        f.write(summary_df.to_markdown())
        f.write("\n")

    # --- Plot 1: growth of $1 ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, r in portfolios.items():
        wealth = (1 + r).cumprod()
        ax.plot(wealth.index, wealth.values, label=name, color=COLORS[name], lw=1.6)
    ax.set_title("Growth of $1 — synthetic 30-year backtest")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "growth_of_dollar.png", dpi=150)
    plt.close(fig)

    # --- Plot 2: drawdowns ---
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for name, r in portfolios.items():
        wealth = (1 + r).cumprod()
        dd = (wealth - wealth.cummax()) / wealth.cummax()
        ax.plot(dd.index, dd.values * 100, label=name, color=COLORS[name], lw=1.3)
    ax.set_title("Drawdown over time")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "drawdowns.png", dpi=150)
    plt.close(fig)

    # --- Plot 3: risk contribution comparison (last window) ---
    from risk_parity import risk_contributions
    cov_full = returns.cov().values * 12
    w_aw = aw_weights
    w_rp = true_risk_parity(cov_full)
    w_ivol = inverse_vol_weights(cov_full)

    rc_aw = risk_contributions(w_aw, cov_full)
    rc_rp = risk_contributions(w_rp, cov_full)
    rc_ivol = risk_contributions(w_ivol, cov_full)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    for ax, (title, w, rc) in zip(
        axes,
        [
            ("All Weather (static)", w_aw, rc_aw),
            ("Inverse Vol", w_ivol, rc_ivol),
            ("True Risk Parity", w_rp, rc_rp),
        ],
    ):
        rc_share = rc / rc.sum() * 100
        w_share = w * 100
        x = np.arange(len(assets))
        ax.bar(x - 0.18, w_share, width=0.35, label="Weight %", color="#C97D3E")
        ax.bar(x + 0.18, rc_share, width=0.35, label="Risk contribution %", color="#4A6FA5")
        ax.set_xticks(x)
        ax.set_xticklabels(assets)
        ax.set_title(title, fontsize=10)
        ax.axhline(100 / len(assets), color="gray", ls="--", lw=0.8)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("% of portfolio")
    fig.suptitle("Capital weight vs. actual risk contribution, by scheme", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "weight_vs_risk_contribution.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved plots + performance summary to {OUT_DIR}/")


if __name__ == "__main__":
    main()
