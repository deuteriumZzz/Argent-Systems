"""Keltner Channel breakout.

Source: classic concept (Chester Keltner's original 10-day moving-average
channel, later popularized in its modern ATR-band form by Linda Bradford
Raschke). Common volatility/trend building block in public crypto bot repos.
Reimplemented from the published formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "trend_keltner_breakout",
    "category": "trend_following",
    "source": "classic concept (Keltner Channel, ATR form popularized by Linda Raschke)",
    "license": "N/A",
    "description": (
        "EMA midline +/- `mult` * ATR bands. Goes long on a close above the "
        "upper band, exits (or shorts if `allow_short`) on a close below "
        "the lower band, otherwise holds the last position."
    ),
    "default_params": {"period": 20, "atr_period": 10, "mult": 2.0, "allow_short": False},
}


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def signals(
    df: pd.DataFrame,
    period: int = 20,
    atr_period: int = 10,
    mult: float = 2.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    mid = close.ewm(span=period, adjust=False).mean()
    atr = _atr(df, atr_period)
    upper = mid + mult * atr
    lower = mid - mult * atr

    short_weight = -1.0 if allow_short else 0.0
    weight = pd.Series(np.nan, index=df.index)
    weight[close > upper] = 1.0
    weight[close < lower] = short_weight
    return weight.ffill().fillna(0.0)
