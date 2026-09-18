"""Supertrend indicator trend following (ATR-based flip line).

Source: classic concept (Olivier Seban's SuperTrend indicator: an ATR-scaled
band around (high+low)/2 that ratchets and flips when price closes through
it). Ubiquitous building block across public crypto bot repos (freqtrade
community strategies, pandas-ta, Jesse example strategies); reimplemented
here from the widely published formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "trend_supertrend",
    "category": "trend_following",
    "source": "classic concept (Olivier Seban's SuperTrend, ATR trend-flip band)",
    "license": "N/A",
    "description": (
        "ATR-scaled band around (high+low)/2 that trails price and flips "
        "direction when close crosses it. Long while the band is below "
        "price (uptrend), flat (or short if `allow_short`) while it is above."
    ),
    "default_params": {"period": 10, "multiplier": 3.0, "allow_short": False},
}


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def signals(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0, allow_short: bool = False, **_
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    atr = _atr(df, period)
    hl2 = (high + low) / 2
    upper_basic = (hl2 + multiplier * atr).to_numpy()
    lower_basic = (hl2 - multiplier * atr).to_numpy()
    close_v = close.to_numpy()

    n = len(df)
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    direction = np.ones(n)  # 1 = uptrend/long, -1 = downtrend/short

    final_upper[0] = upper_basic[0]
    final_lower[0] = lower_basic[0]

    for i in range(1, n):
        final_upper[i] = (
            upper_basic[i]
            if (upper_basic[i] < final_upper[i - 1] or close_v[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower_basic[i]
            if (lower_basic[i] > final_lower[i - 1] or close_v[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        if direction[i - 1] == 1:
            direction[i] = -1 if close_v[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close_v[i] > final_upper[i] else -1

    short_weight = -1.0 if allow_short else 0.0
    return pd.Series(direction, index=df.index).map({1: 1.0, -1: short_weight})
