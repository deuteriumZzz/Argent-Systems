"""Commodity Channel Index (CCI) mean reversion.

Source: classic technical-analysis concept, developed by Donald Lambert
(1980, Commodities magazine). A staple oscillator in public crypto bot
repos for fading extremes away from the typical-price moving average.
Reimplemented from the published formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "reversion_cci",
    "category": "mean_reversion",
    "source": "classic concept (Donald Lambert's CCI)",
    "license": "N/A",
    "description": (
        "CCI = (typical price - SMA(typical, period)) / (0.015 * mean "
        "absolute deviation). Goes long when CCI < -`threshold` (deeply "
        "below the mean), exits/shorts when CCI > `threshold`."
    ),
    "default_params": {"period": 20, "threshold": 100, "allow_short": False},
}


def signals(
    df: pd.DataFrame, period: int = 20, threshold: float = 100, allow_short: bool = False, **_
) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (typical - sma) / (0.015 * mad.replace(0, np.nan))

    long_entries = cci < -threshold
    short_entries = cci > threshold

    weight = pd.Series(np.nan, index=df.index)
    weight[long_entries] = 1.0
    weight[short_entries] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
