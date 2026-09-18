"""Elder Force Index (dual-period trend + trigger) system.

Source: classic concept (Alexander Elder's Force Index, from "Trading for
a Living", 1993; documented formula as used by StockCharts/Investopedia).
Force Index = bar-to-bar close change * volume — it is strong when price
moves a lot on heavy volume, weak when it moves on thin volume. Elder's own
system, reproduced here, smooths Force Index at two horizons: a slow EMA
(13) sets the trend context, a fast EMA (2) times entries within it.
Reimplemented from the publicly documented formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "volume_elder_force_index",
    "category": "trend_following",
    "source": "classic concept (Alexander Elder's Force Index trading system)",
    "license": "N/A",
    "description": (
        "Force Index = (close - close.shift(1)) * volume. `slow_period`-EMA "
        "of it sets trend direction; long entry triggers when the "
        "`fast_period`-EMA crosses up through zero while the slow EMA is "
        "positive (a brief volume-backed pullback resolving back with the "
        "uptrend). Exits (flattens) when the slow EMA trend flips sign. "
        "Mirrors the same logic short-side if `allow_short`."
    ),
    "default_params": {"fast_period": 2, "slow_period": 13, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    fast_period: int = 2,
    slow_period: int = 13,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close, volume = df["close"], df["volume"]
    force = close.diff() * volume

    fi_fast = force.ewm(span=fast_period, adjust=False).mean()
    fi_slow = force.ewm(span=slow_period, adjust=False).mean()

    trend_up = fi_slow > 0
    trend_down = fi_slow < 0

    long_entry = trend_up & (fi_fast > 0) & (fi_fast.shift(1) <= 0)
    short_entry = trend_down & (fi_fast < 0) & (fi_fast.shift(1) >= 0)
    long_exit = ~trend_up

    weight = pd.Series(np.nan, index=df.index)
    # exit set first so a same-bar entry (below) can override the flatten
    weight[long_exit] = 0.0
    if allow_short:
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)
