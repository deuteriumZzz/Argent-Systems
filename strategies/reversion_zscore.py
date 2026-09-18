"""Rolling z-score mean reversion on price.

Standardizes close price against its own rolling mean/std (a rolling
z-score) and fades extreme deviations back toward zero. This is the
textbook "statistical arbitrage" building block behind countless single-
asset and pairs mean-reversion bots (the single-asset case used here is
the simplest form, with no cointegrated second leg).

Source: classic quantitative-trading concept (rolling z-score / Bollinger's
underlying statistic). Reimplemented from the standard formula, no code
borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "reversion_zscore",
    "category": "mean_reversion",
    "source": "classic concept (rolling z-score statistical arbitrage)",
    "license": "N/A",
    "description": (
        "Computes a rolling z-score of close price over `period` bars. "
        "Goes long when z-score < -`entry_z`, shorts (if `allow_short`) "
        "when z-score > `entry_z`, and flattens once |z-score| < `exit_z`."
    ),
    "default_params": {
        "period": 30,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    period: int = 30,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    zscore = (close - mean) / std.replace(0, np.nan)

    long_entries = zscore < -entry_z
    short_entries = zscore > entry_z
    flat = zscore.abs() < exit_z

    weight = pd.Series(np.nan, index=df.index)
    weight[flat] = 0.0
    weight[long_entries] = 1.0
    weight[short_entries] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
