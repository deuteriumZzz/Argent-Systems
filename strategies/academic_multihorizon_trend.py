"""Multi-horizon trend ensemble (Hurst, Ooi & Pedersen, 2017).

Source: Hurst, B., Ooi, Y.H., Pedersen, L.H. (2017), "A Century of Evidence
on Trend-Following Investing", Journal of Portfolio Management 44(1), and
its AQR predecessor working paper of the same title
(https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing).
Their trend signal blends multiple lookback horizons (roughly 1, 3 and 12
months) rather than committing to one - short lookbacks catch fast trends,
long ones catch slow ones, and averaging the horizon signals produces a
smoother position that only reaches full size when trends agree across
speeds. Reimplemented here for a single asset (no code reused): average of
sign(trailing return) across `lookbacks`, vol-targeted.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_multihorizon_trend",
    "category": "trend_following",
    "source": "Hurst, Ooi & Pedersen (2017), 'A Century of Evidence on Trend-Following Investing', JPM 44(1)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "For each lookback in `lookbacks` (default ~1/3/12-month bar "
        "equivalents), takes sign(trailing return). Averages the per-"
        "horizon signals into one score in [-1, 1] - full size only when "
        "fast and slow trends agree, damped when they conflict - then "
        "scales to `target_vol` using trailing realized volatility."
    ),
    "default_params": {
        "lookbacks": (21, 63, 252),
        "vol_lookback": 20,
        "target_vol": 0.02,
        "max_leverage": 1.0,
    },
}


def signals(
    df: pd.DataFrame,
    lookbacks: tuple[int, ...] = (21, 63, 252),
    vol_lookback: int = 20,
    target_vol: float = 0.02,
    max_leverage: float = 1.0,
    **_,
) -> pd.Series:
    close = df["close"]
    per_horizon = [np.sign(close.pct_change(lb)) for lb in lookbacks]
    ensemble = sum(per_horizon) / len(per_horizon)

    bar_return = close.pct_change()
    realized_vol = bar_return.rolling(vol_lookback).std()
    size = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan).clip(upper=max_leverage)

    weight = (ensemble * size).clip(-1.0, 1.0)
    return weight.fillna(0.0)
