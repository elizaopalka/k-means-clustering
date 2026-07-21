"""
risk_parity.py

Portfolio weighting schemes, implemented from scratch with numpy.

    inverse_vol_weights    -- naive risk parity: weight inversely
                               proportional to each asset's own volatility
                               (ignores correlation structure)

    true_risk_parity       -- equal *risk contribution* weighting, solved
                               via the cyclical fixed-point iteration from
                               Maillard, Roncalli & Teiletche (2010): each
                               asset's contribution to total portfolio
                               variance is equalized, not just its raw vol.
                               This is what "risk parity" means in practice
                               -- it accounts for how assets move together,
                               not just how much each one moves alone.

Risk contribution of asset i:
    RC_i = w_i * (Sigma @ w)_i / sqrt(w' Sigma w)

Target: RC_i / sum(RC) == 1/N for all i.
"""

import numpy as np


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.diag(cov))
    w = 1.0 / vol
    return w / w.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    port_var = w @ cov @ w
    marginal = cov @ w
    rc = w * marginal / np.sqrt(port_var)
    return rc


def true_risk_parity(cov: np.ndarray, max_iter: int = 500, tol: float = 1e-10) -> np.ndarray:
    n = cov.shape[0]
    w = np.ones(n) / n  # start from equal weight

    target = 1.0 / n
    prev_gap = np.inf

    for _ in range(max_iter):
        rc = risk_contributions(w, cov)
        rc_share = rc / rc.sum()

        # Safeguard: some assets (e.g. long bonds hedging equities) can have
        # a *negative* marginal risk contribution -- they reduce portfolio
        # variance rather than add to it. A negative share under the sqrt
        # fixed-point update is undefined, so floor it above zero; this
        # damps convergence slightly for strong diversifiers but keeps the
        # iteration numerically stable and weights long-only.
        rc_share_safe = np.clip(rc_share, 1e-4, None)

        w = w * np.sqrt(target / rc_share_safe)
        w = np.clip(w, 1e-6, None)
        w = w / w.sum()

        gap = np.max(np.abs(rc_share - target))
        if gap < tol or abs(prev_gap - gap) < tol:
            break
        prev_gap = gap

    return w


def equal_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    return np.ones(n) / n


def all_weather_weights(asset_order: list) -> np.ndarray:
    """
    Classic Ray Dalio All Weather static allocation:
    30% equities, 40% long-term bonds, 15% intermediate bonds,
    7.5% gold, 7.5% commodities.
    """
    target = {"EQ": 0.30, "LTB": 0.40, "ITB": 0.15, "GLD": 0.075, "CMD": 0.075}
    return np.array([target[a] for a in asset_order])
