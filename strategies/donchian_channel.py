"""Donchian channel breakout (Turtle Trading style).

Source: classic technical-analysis concept popularized by Richard Donchian
and the 1980s "Turtle Traders"; reimplemented from the published rules, no
code borrowed. A staple trend-following building block in public crypto bot
repos.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "donchian_channel",
    "category": "trend_following",
    "source": "classic concept (Richard Donchian / Turtle Traders)",
    "license": "N/A",
    "description": (
        "Long breakout system: enters when price closes above the highest "
        "high of the past `entry_period` bars, exits when price closes "
        "below the lowest low of the past `exit_period` bars."
    ),
    "default_params": {"entry_period": 20, "exit_period": 10},
}


def signals(df: pd.DataFrame, entry_period: int = 20, exit_period: int = 10, **_) -> pd.Series:
    close = df["close"]
    entry_high = df["high"].rolling(entry_period).max().shift(1)
    exit_low = df["low"].rolling(exit_period).min().shift(1)

    entries = close > entry_high
    exits = close < exit_low

    weight = pd.Series(np.nan, index=df.index)
    weight[entries] = 1.0
    weight[exits] = 0.0
    return weight.ffill().fillna(0.0)
