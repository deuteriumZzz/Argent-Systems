"""HalfTrend, simplified (niche indicator originating in TradingView Pine).

Source: HalfTrend, a widely-copied but non-canonical trend-flip indicator
originally published as a TradingView Pine Script by user "everget"
(https://www.tradingview.com/script/U1SJ8ubc-HalfTrend/) and ported into a
small, obscure Python repo at ryu878/halftrend_python
(https://github.com/ryu878/halftrend_python). This is a SIMPLIFIED
reimplementation of the documented two-state mechanic (an SMA-of-high/low
trailing channel that flips trend on a breakout) — not a line-by-line port
of the Pine Script or the Python fork; no code copied from either.
License: source has no clearly stated license; treated as a documented
mechanic only, reimplemented independently from a written description.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "trend_halftrend",
    "category": "trend_following",
    "source": (
        "niche indicator (HalfTrend, TradingView Pine Script by 'everget'; "
        "ported at ryu878/halftrend_python) — simplified reimplementation"
    ),
    "license": "unclear/none on source — reimplemented independently from documented mechanic",
    "description": (
        "SMA(high, amplitude) / SMA(low, amplitude) trailing channel. "
        "Uptrend's reference line ratchets up to the max of the current low "
        "SMA and its own last value; downtrend's mirrors on the high SMA. "
        "Flips (and resets) when price closes through the reference."
    ),
    "default_params": {"amplitude": 4, "allow_short": False},
}


def signals(df: pd.DataFrame, amplitude: int = 4, allow_short: bool = False, **_) -> pd.Series:
    ma_high = df["high"].rolling(amplitude).mean().to_numpy()
    ma_low = df["low"].rolling(amplitude).mean().to_numpy()
    close_v = df["close"].to_numpy()
    n = len(df)

    trend = np.ones(n)
    ref = np.zeros(n)
    ref[0] = ma_low[0] if not np.isnan(ma_low[0]) else close_v[0]

    for i in range(1, n):
        if trend[i - 1] == 1:
            candidate = (
                min(ma_low[i], ref[i - 1]) if not np.isnan(ma_low[i]) else ref[i - 1]
            )
            if close_v[i] < candidate:
                trend[i] = -1
                ref[i] = ma_high[i] if not np.isnan(ma_high[i]) else candidate
            else:
                trend[i] = 1
                ref[i] = candidate
        else:
            candidate = (
                max(ma_high[i], ref[i - 1]) if not np.isnan(ma_high[i]) else ref[i - 1]
            )
            if close_v[i] > candidate:
                trend[i] = 1
                ref[i] = ma_low[i] if not np.isnan(ma_low[i]) else candidate
            else:
                trend[i] = -1
                ref[i] = candidate

    short_weight = -1.0 if allow_short else 0.0
    return pd.Series(trend, index=df.index).map({1: 1.0, -1: short_weight})
