"""Bollinger Bands breakout.

Source: classic technical-analysis concept (John Bollinger, 1980s);
reimplemented from the published formula, no code borrowed. One of the most
common volatility-based building blocks in public crypto bot repos.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "bollinger_breakout",
    "category": "volatility_breakout",
    "source": "classic concept (John Bollinger's Bollinger Bands)",
    "license": "N/A",
    "description": (
        "Goes long when price closes above the upper band (momentum "
        "breakout), exits (or shorts if `allow_short`) when price closes "
        "below the lower band, otherwise holds the last position."
    ),
    "default_params": {"period": 20, "num_std": 2.0, "allow_short": False},
}


def signals(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0, allow_short: bool = False, **_
) -> pd.Series:
    close = df["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std

    short_weight = -1.0 if allow_short else 0.0
    weight = pd.Series(np.nan, index=df.index)
    weight[close > upper] = 1.0
    weight[close < lower] = short_weight
    return weight.ffill().fillna(0.0)
