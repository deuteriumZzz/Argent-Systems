"""Fibonacci-spaced grid bot.

Source: mechanic documented by Finandy's "Fibonacci grid (levels)" order
grid — instead of equally-spaced levels, grid lines sit at Fibonacci ratios
between the range boundaries, clustering trading around the golden-ratio
zone rather than uniformly across the range. See:
https://docs.finandy.com/trading/create-order-grid/fibonacci-grid-levels
Reimplemented from the publicly documented mechanic, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

_FIB_RATIOS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

META = {
    "name": "grid_fibonacci",
    "category": "grid",
    "source": "Finandy Fibonacci grid (levels)",
    "license": "N/A",
    "description": (
        "Same [lower, upper] range as a static grid bot, but levels sit at "
        "Fibonacci ratios (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0) instead "
        "of equal spacing, so trading is denser around the middle of the "
        "range. Buys an equal-size unit at each level, sells it when price "
        "climbs back to the next ratio up."
    ),
    "default_params": {"range_pct": 0.30},
}


def signals(df: pd.DataFrame, range_pct: float = 0.30, **_) -> pd.Series:
    close = df["close"]
    start_price = close.iloc[0]
    lower = start_price * (1 - range_pct / 2)
    upper = start_price * (1 + range_pct / 2)
    span = upper - lower
    levels = [lower + r * span for r in _FIB_RATIOS]
    num_levels = len(levels) - 1
    unit = 1.0 / num_levels

    weight = pd.Series(0.0, index=df.index)
    held_levels: set[int] = set()
    current_weight = 0.0

    prices = close.to_numpy()
    for i, price in enumerate(prices):
        for level_idx in range(num_levels):
            level_price = levels[level_idx]
            if level_idx not in held_levels and lower <= price <= level_price:
                held_levels.add(level_idx)
                current_weight = min(1.0, current_weight + unit)
        for level_idx in list(held_levels):
            if price >= levels[level_idx + 1]:
                held_levels.discard(level_idx)
                current_weight = max(0.0, current_weight - unit)
        weight.iloc[i] = current_weight
    return weight
