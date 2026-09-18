"""SSL Channel trend flip (niche freqtrade-community indicator).

Source: found in small/niche open-source freqtrade-community repos rather
than textbook TA literature — the `SSLChannels` helper in
freqtrade/technical (https://github.com/freqtrade/technical) and its use in
froggleston/cryptofrog-strategies' custom_indicators.py
(https://github.com/froggleston/cryptofrog-strategies). The documented
mechanic: SMAs of high/low form a channel, and the "current" band flips
whenever close crosses the opposite SMA. Reimplemented here independently
from that documented mechanic only — no code copied from either repo.
License: GPL-3.0 (source repos); this reimplementation is original, formula
only, no code copied.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "trend_ssl_channel",
    "category": "trend_following",
    "source": (
        "niche freqtrade-community indicator (freqtrade/technical "
        "SSLChannels; also used in froggleston/cryptofrog-strategies)"
    ),
    "license": "GPL-3.0 (source repos) — reimplemented from documented formula, no code copied",
    "description": (
        "SMA(high, period) / SMA(low, period) channel. Direction flips to "
        "long when close closes above SMA(high), and to short when close "
        "closes below SMA(low); holds the last direction in between."
    ),
    "default_params": {"period": 10, "allow_short": False},
}


def signals(df: pd.DataFrame, period: int = 10, allow_short: bool = False, **_) -> pd.Series:
    close = df["close"]
    sma_high = df["high"].rolling(period).mean()
    sma_low = df["low"].rolling(period).mean()

    direction = pd.Series(np.nan, index=df.index)
    direction[close > sma_high] = 1.0
    direction[close < sma_low] = -1.0
    direction = direction.ffill().fillna(0.0)

    if not allow_short:
        direction = direction.clip(lower=0.0)
    return direction
