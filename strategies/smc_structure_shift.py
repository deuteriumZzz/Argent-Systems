"""Break of Structure (BOS) / Change of Character (CHoCH): trading market
structure shifts defined by sequences of swing highs and swing lows.

Source: ICT / Smart Money Concepts methodology. Swing points detected via
a simple fractal rule (a bar is a swing high/low if it's the extreme of a
symmetric window around it) since ICT doesn't formally specify one —
different educators use different swing-detection rules. This is one
reasonable, backtestable interpretation, not a canonical algorithm.
License: N/A (public trading concept, reimplemented).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "smc_structure_shift",
    "category": "smart_money_concepts",
    "source": "ICT / Smart Money Concepts, 'Break of Structure' / 'Change of Character'",
    "license": "N/A (public trading concept, reimplemented)",
    "description": (
        "Tracks swing highs/lows via a fractal rule. A close above the last "
        "confirmed swing high goes/stays long (continuation = BOS, or "
        "reversal from a downtrend = CHoCH); a close below the last "
        "confirmed swing low goes/stays short. Mirrored logic both ways."
    ),
    "default_params": {"swing_window": 5},
}


def signals(df: pd.DataFrame, swing_window: int = 5, **_) -> pd.Series:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(close)
    weight = np.zeros(n)
    w = swing_window

    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        if high[i] == high[i - w : i + w + 1].max():
            is_swing_high[i] = True
        if low[i] == low[i - w : i + w + 1].min():
            is_swing_low[i] = True

    last_swing_high = None
    last_swing_low = None
    current_weight = 0.0

    for i in range(n):
        # a swing at bar (i - w) only becomes knowable once w bars past it
        # exist (it needed future bars to confirm it was the local extreme)
        confirm_idx = i - w
        if confirm_idx >= 0:
            if is_swing_high[confirm_idx]:
                last_swing_high = high[confirm_idx]
            if is_swing_low[confirm_idx]:
                last_swing_low = low[confirm_idx]

        price = close[i]
        if last_swing_high is not None and price > last_swing_high:
            current_weight = 1.0
        if last_swing_low is not None and price < last_swing_low:
            current_weight = -1.0

        weight[i] = current_weight

    return pd.Series(weight, index=df.index)
