"""Vortex Indicator crossover.

Source: classic concept (Vortex Indicator, Etienne Botes & Douglas Siepman,
"Technical Analysis of Stocks & Commodities" magazine, 2010). A less common
but publicly documented trend-following indicator occasionally seen in
crypto TA toolkits (e.g. pandas-ta's `vortex`). Reimplemented from the
published VI+/VI- formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "trend_vortex",
    "category": "trend_following",
    "source": "classic concept (Botes & Siepman's Vortex Indicator)",
    "license": "N/A",
    "description": (
        "Long while VI+ (upward vortex movement) is above VI- (downward "
        "vortex movement) over `period` bars, flat (or short if "
        "`allow_short`) otherwise."
    ),
    "default_params": {"period": 14, "allow_short": False},
}


def signals(df: pd.DataFrame, period: int = 14, allow_short: bool = False, **_) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    prev_low = low.shift(1)
    prev_high = high.shift(1)

    vm_plus = (high - prev_low).abs()
    vm_minus = (low - prev_high).abs()
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    tr_sum = tr.rolling(period).sum().replace(0, np.nan)
    vi_plus = vm_plus.rolling(period).sum() / tr_sum
    vi_minus = vm_minus.rolling(period).sum() / tr_sum

    bullish = vi_plus > vi_minus
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})
