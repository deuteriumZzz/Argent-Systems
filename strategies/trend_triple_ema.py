"""Triple EMA crossover (3-EMA ribbon alignment).

Source: classic concept (three exponential moving averages of increasing
length used as a "ribbon"; long only while all three are stacked bullishly).
A very common building block across public crypto/freqtrade-style strategy
repos. Reimplemented from the standard EMA formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "trend_triple_ema",
    "category": "trend_following",
    "source": "classic concept (3-EMA ribbon / triple EMA crossover)",
    "license": "N/A",
    "description": (
        "Long while fast EMA > mid EMA > slow EMA (bullish ribbon stack), "
        "flat (or short if `allow_short`) while the stack is bearish, "
        "otherwise flat."
    ),
    "default_params": {"fast": 5, "mid": 10, "slow": 20, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    fast: int = 5,
    mid: int = 10,
    slow: int = 20,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_mid = close.ewm(span=mid, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    bullish = (ema_fast > ema_mid) & (ema_mid > ema_slow)
    bearish = (ema_fast < ema_mid) & (ema_mid < ema_slow)

    short_weight = -1.0 if allow_short else 0.0
    weight = pd.Series(0.0, index=df.index)
    weight[bullish] = 1.0
    weight[bearish] = short_weight
    return weight
