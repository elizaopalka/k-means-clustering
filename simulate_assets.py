"""
simulate_assets.py

Generates synthetic monthly return series for the four asset classes used
in Ray Dalio's All Weather portfolio, calibrated to realistic long-run
return/volatility/correlation characteristics (not fit to any specific
historical window -- these are standard textbook parameters for each
asset class's behavior across growth/inflation regimes).

Asset classes:
    EQ   - Equities (e.g. S&P 500 total return)
    LTB  - Long-term government bonds (e.g. 20-30yr Treasuries)
    ITB  - Intermediate-term government bonds (e.g. 7-10yr Treasuries)
    GLD  - Gold
    CMD  - Broad commodities

Regime structure: returns are driven by two latent macro factors --
growth and inflation surprises -- which is the actual economic logic
behind the All Weather portfolio (each asset class is chosen because of
how it responds to the four growth/inflation quadrants). This is why
stocks and bonds are negatively correlated in most regimes but not all.

Output: data/synthetic_asset_returns.csv (monthly returns, one column
per asset class)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 7
N_MONTHS = 30 * 12  # 30 years of monthly data

# Annualized target return / vol per asset class (long-run, illustrative)
ASSET_PARAMS = {
    "EQ":  {"mu": 0.095, "sigma": 0.16},
    "LTB": {"mu": 0.045, "sigma": 0.11},
    "ITB": {"mu": 0.038, "sigma": 0.05},
    "GLD": {"mu": 0.055, "sigma": 0.15},
    "CMD": {"mu": 0.050, "sigma": 0.18},
}

# Factor loadings on [growth_surprise, inflation_surprise]
# This encodes the actual All Weather logic:
#   equities love growth, hate nothing much
#   long bonds hate inflation, love falling growth (flight to safety)
#   gold hates real yields / loves inflation
#   commodities love inflation and growth both
FACTOR_LOADINGS = {
    "EQ":  np.array([0.85, -0.10]),
    "LTB": np.array([-0.35, -0.65]),
    "ITB": np.array([-0.15, -0.35]),
    "GLD": np.array([0.05, 0.55]),
    "CMD": np.array([0.35, 0.60]),
}

FACTOR_VOL = np.array([0.11, 0.07])  # annualized vol of growth, inflation surprises


def simulate(seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = 1 / 12

    factors = rng.normal(0, FACTOR_VOL * np.sqrt(dt), size=(N_MONTHS, 2))

    assets = list(ASSET_PARAMS.keys())
    returns = np.zeros((N_MONTHS, len(assets)))

    for j, name in enumerate(assets):
        mu = ASSET_PARAMS[name]["mu"]
        sigma = ASSET_PARAMS[name]["sigma"]
        loading = FACTOR_LOADINGS[name]

        factor_component = factors @ loading
        # scale idiosyncratic noise so total vol matches target sigma
        explained_var = np.var(factor_component) * 12  # annualized
        idio_var = max(sigma ** 2 - explained_var, 0.15 * sigma ** 2)
        idio_sigma_m = np.sqrt(idio_var * dt)

        idio = rng.normal(0, idio_sigma_m, size=N_MONTHS)
        returns[:, j] = mu * dt + factor_component + idio

    dates = pd.date_range("1995-01-31", periods=N_MONTHS, freq="ME")
    df = pd.DataFrame(returns, columns=assets, index=dates)
    df.index.name = "date"
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(exist_ok=True)
    df = simulate()
    out_path = out_dir / "synthetic_asset_returns.csv"
    df.to_csv(out_path)

    ann_stats = pd.DataFrame({
        "ann_return": df.mean() * 12,
        "ann_vol": df.std() * np.sqrt(12),
    })
    print(ann_stats.round(4))
    print(f"\nWrote {df.shape[0]} months x {df.shape[1]} assets -> {out_path}")
