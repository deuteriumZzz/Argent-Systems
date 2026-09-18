"""Chandelier Exit trend flip (ATR trailing stop used as a signal).

Source: classic concept (Chandelier Exit, Chuck LeBeau) — an ATR-scaled
trailing stop from the highest high (for longs) / lowest low (for shorts)
over a lookback window. Normally used only as a stop-loss, but also seen
repurposed as a standalone trend-flip signal in small/niche open crypto
repos, e.g. botradingblog1/python-algorithmic-trading's
"Crypto Chandelier Exit.ipynb" notebook. Reimplemented here from the
publicly documented ATR-trailing-stop mechanic, no code borrowed.
License: N/A (concept); example repo's own license is unclear, no code
copied from it either way.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "trend_chandelier_exit",
    "category": "trend_following",
    "source": (
        "classic concept (Chuck LeBeau's Chandelier Exit); niche crypto "
        "port example: botradingblog1/python-algorithmic-trading"
    ),
    "license": "N/A (concept)",
    "description": (
        "ATR-scaled trailing stop off the rolling highest-high / lowest-low "
        "over `period` bars, repurposed as a trend-flip signal: long while "
        "price holds above its trailing long-stop, short (if `allow_short`) "
        "once it closes below, and vice versa."
    ),
    "default_params": {"period": 22, "mult": 3.0, "allow_short": False},
}


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def signals(
    df: pd.DataFrame, period: int = 22, mult: float = 3.0, allow_short: bool = False, **_
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    atr = _atr(df, period)
    long_stop_basic = (high.rolling(period).max() - mult * atr).to_numpy()
    short_stop_basic = (low.rolling(period).min() + mult * atr).to_numpy()
    close_v = close.to_numpy()

    n = len(df)
    long_stop = np.zeros(n)
    short_stop = np.zeros(n)
    direction = np.ones(n)

    long_stop[0] = long_stop_basic[0]
    short_stop[0] = short_stop_basic[0]

    for i in range(1, n):
        long_stop[i] = (
            max(long_stop_basic[i], long_stop[i - 1])
            if close_v[i - 1] > long_stop[i - 1]
            else long_stop_basic[i]
        )
        short_stop[i] = (
            min(short_stop_basic[i], short_stop[i - 1])
            if close_v[i - 1] < short_stop[i - 1]
            else short_stop_basic[i]
        )

        if direction[i - 1] == -1 and close_v[i] > short_stop[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and close_v[i] < long_stop[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    short_weight = -1.0 if allow_short else 0.0
    return pd.Series(direction, index=df.index).map({1: 1.0, -1: short_weight})
