"""Stochastic Oscillator (%K/%D) mean reversion.

Source: classic technical-analysis concept, developed by George Lane in the
1950s. One of the most widely used bounded oscillators in public crypto bot
repos for overbought/oversold reversion signals. Reimplemented from the
published formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "reversion_stochastic",
    "category": "mean_reversion",
    "source": "classic concept (George Lane's Stochastic Oscillator)",
    "license": "N/A",
    "description": (
        "%K = position of close within the high/low range over `k_period`, "
        "%D = SMA(%K, d_period). Goes long when both %K and %D are below "
        "`oversold`, exits/shorts when both are above `overbought`."
    ),
    "default_params": {
        "k_period": 14,
        "d_period": 3,
        "oversold": 20,
        "overbought": 80,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    oversold: float = 20,
    overbought: float = 80,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    span = (highest_high - lowest_low).replace(0, np.nan)
    percent_k = 100 * (close - lowest_low) / span
    percent_d = percent_k.rolling(d_period).mean()

    long_entries = (percent_k < oversold) & (percent_d < oversold)
    short_entries = (percent_k > overbought) & (percent_d > overbought)

    weight = pd.Series(np.nan, index=df.index)
    weight[long_entries] = 1.0
    weight[short_entries] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
