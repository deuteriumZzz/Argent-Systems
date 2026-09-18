"""Accumulation/Distribution Line (A/D Line) trend-following crossover.

Source: classic concept (Marc Chaikin's Accumulation/Distribution Line,
1970s; documented formula as used by StockCharts/Investopedia). Unlike the
Chaikin Money Flow used elsewhere in this repo (a *rolling*, volume-
normalized ratio), the A/D Line is a *cumulative running total* of the
money-flow volume — closer in spirit to OBV but weighted by where the close
sits within the bar's range instead of just its sign. This strategy treats
the level of the A/D Line itself as a trend signal via a fast/slow EMA
crossover on the line. Reimplemented from the publicly documented formula,
no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "volume_ad_line_trend",
    "category": "trend_following",
    "source": "classic concept (Marc Chaikin's Accumulation/Distribution Line)",
    "license": "N/A",
    "description": (
        "Money Flow Multiplier = ((close-low)-(high-close))/(high-low), "
        "times volume, cumulatively summed into the A/D Line. Long while "
        "the A/D Line's `fast_period`-bar EMA is above its `slow_period`-bar "
        "EMA (accumulation trend), flat (or short if `allow_short`) while "
        "below."
    ),
    "default_params": {"fast_period": 10, "slow_period": 30, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 30,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = (mfm * volume).fillna(0.0)
    adl = mfv.cumsum()

    fast_ema = adl.ewm(span=fast_period, adjust=False).mean()
    slow_ema = adl.ewm(span=slow_period, adjust=False).mean()

    short_weight = -1.0 if allow_short else 0.0
    return (fast_ema > slow_ema).map({True: 1.0, False: short_weight})
