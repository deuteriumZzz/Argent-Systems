"""Optimal Trade Entry (OTE): enter in the direction of a fresh structure
break once price retraces into the 61.8%-79% Fibonacci zone of the swing
that caused it.

Source: ICT / Smart Money Concepts methodology. Builds on the same
swing/structure-break detection as smc_structure_shift.py (reimplemented
standalone here per this repo's no-cross-imports rule), adding the
Fibonacci entry-zone refinement ICT calls "Optimal Trade Entry". One
reasonable, backtestable interpretation, not a canonical algorithm.
License: N/A (public trading concept, reimplemented).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "smc_optimal_trade_entry",
    "category": "smart_money_concepts",
    "source": "ICT / Smart Money Concepts, 'Optimal Trade Entry'",
    "license": "N/A (public trading concept, reimplemented)",
    "description": (
        "After a fresh break above the last swing high, waits for price to "
        "pull back into the 61.8-79% Fibonacci retracement zone of the "
        "impulse leg before going long; mirrored for a break below the "
        "last swing low. Invalidated (flat) if price retraces past the "
        "start of that impulse leg."
    ),
    "default_params": {"swing_window": 5, "fib_shallow": 0.618, "fib_deep": 0.79},
}


def signals(
    df: pd.DataFrame, swing_window: int = 5, fib_shallow: float = 0.618, fib_deep: float = 0.79, **_
) -> pd.Series:
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
    state = "idle"  # idle | awaiting_bull_ote | awaiting_bear_ote
    leg_start = None  # price at the start of the impulse leg (the pre-break swing point)
    leg_extreme = None  # running peak (bull) / trough (bear) reached since the break
    current_weight = 0.0

    for i in range(n):
        confirm_idx = i - w
        if confirm_idx >= 0:
            if is_swing_high[confirm_idx]:
                last_swing_high = high[confirm_idx]
            if is_swing_low[confirm_idx]:
                last_swing_low = low[confirm_idx]

        price = close[i]

        if last_swing_high is not None and price > last_swing_high and state != "awaiting_bull_ote":
            state = "awaiting_bull_ote"
            leg_start = last_swing_low if last_swing_low is not None else price
            leg_extreme = price
        if last_swing_low is not None and price < last_swing_low and state != "awaiting_bear_ote":
            state = "awaiting_bear_ote"
            leg_start = last_swing_high if last_swing_high is not None else price
            leg_extreme = price

        if state == "awaiting_bull_ote":
            leg_extreme = max(leg_extreme, price)
            leg_range = leg_extreme - leg_start
            zone_hi = leg_extreme - fib_shallow * leg_range
            zone_lo = leg_extreme - fib_deep * leg_range
            if price < leg_start:
                state, current_weight = "idle", 0.0
            elif zone_lo <= price <= zone_hi:
                current_weight = 1.0

        if state == "awaiting_bear_ote":
            leg_extreme = min(leg_extreme, price)
            leg_range = leg_start - leg_extreme
            zone_lo = leg_extreme + fib_shallow * leg_range
            zone_hi = leg_extreme + fib_deep * leg_range
            if price > leg_start:
                state, current_weight = "idle", 0.0
            elif zone_lo <= price <= zone_hi:
                current_weight = -1.0

        weight[i] = current_weight

    return pd.Series(weight, index=df.index)
