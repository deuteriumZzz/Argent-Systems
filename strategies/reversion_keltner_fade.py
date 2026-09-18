"""Keltner Channel mean reversion (fades band touches).

Unlike `trend_keltner_breakout.py` (which trades a close beyond the bands
as continuation), this fades the same channel: touching or piercing a band
is read as an overextension, and the trade targets a snap back to the EMA
midline.

Source: classic concept (Chester Keltner's original moving-average channel,
ATR-band form popularized by Linda Bradford Raschke). Reimplemented from
the published formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "reversion_keltner_fade",
    "category": "mean_reversion",
    "source": "classic concept (Keltner Channel, ATR form popularized by Linda Raschke)",
    "license": "N/A",
    "description": (
        "EMA midline +/- `mult` * ATR bands. Goes long when close dips "
        "below the lower band (fade the overextension), exits once close "
        "reverts back above the midline. Mirrors short-side if `allow_short`."
    ),
    "default_params": {
        "period": 20,
        "atr_period": 10,
        "mult": 2.0,
        "allow_short": False,
    },
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

    long_entries = close < lower
    short_entries = close > upper
    flat = close.sub(mid).abs() < atr * 0.1

    weight = pd.Series(np.nan, index=df.index)
    weight[flat] = 0.0
    weight[long_entries] = 1.0
    weight[short_entries] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
