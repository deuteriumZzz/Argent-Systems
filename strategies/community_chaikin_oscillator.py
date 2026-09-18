"""Chaikin Oscillator — MACD of the Accumulation/Distribution Line.

Source: classic concept (Marc Chaikin's Chaikin Oscillator, 1980s;
documented formula as used by StockCharts/Investopedia). Distinct from the
plain Chaikin Money Flow used elsewhere in this repo (a rolling ratio) and
from the raw A/D Line trend crossover also in this repo (an EMA crossover
directly on the line): the Chaikin Oscillator is specifically the
*difference* of a fast and slow EMA of the A/D Line, i.e. MACD applied to
volume-weighted accumulation instead of to price. Reimplemented from the
publicly documented formula, no code borrowed. Documented widely on
TradingView community script listings as well as classic TA references.
License: N/A (public indicator formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "community_chaikin_oscillator",
    "category": "trend_following",
    "source": "classic concept (Marc Chaikin's Chaikin Oscillator)",
    "license": "N/A (public indicator formula)",
    "description": (
        "A/D Line = cumulative sum of money-flow-multiplier * volume. "
        "Chaikin Oscillator = EMA(A/D Line, fast_period) - EMA(A/D Line, "
        "slow_period). Long while the oscillator is above `entry_threshold`, "
        "flat (or short if `allow_short`) while below -`entry_threshold`, "
        "holds previous weight in between."
    ),
    "default_params": {
        "fast_period": 3,
        "slow_period": 10,
        "entry_threshold": 0.0,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    fast_period: int = 3,
    slow_period: int = 10,
    entry_threshold: float = 0.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    adl = (mfm * volume).fillna(0.0).cumsum()

    oscillator = adl.ewm(span=fast_period, adjust=False).mean() - adl.ewm(span=slow_period, adjust=False).mean()

    weight = pd.Series(np.nan, index=df.index)
    weight[oscillator > entry_threshold] = 1.0
    weight[oscillator < -entry_threshold] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
