"""Inverse (short) grid bot for range-bound markets.

Source: mechanic documented by Pionex Futures Grid Bot's "Short" mode — a
short grid opens/adds short exposure at grid levels above the current price
and covers as price falls back through them, mirroring a normal long grid's
buy-low/sell-high but flipped for a range-bound-or-bearish outlook. See:
https://www.pionex.com/blog/pionex_short_futures_grid/
https://support.pionex.com/hc/en-us/articles/45343668185113-Futures-Grid-Bot
Reimplemented from the publicly documented mechanic, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "grid_inverse_short",
    "category": "grid",
    "source": "Pionex Futures Grid Bot (Short mode)",
    "license": "N/A",
    "description": (
        "Mirror image of a static long grid: splits [lower, upper] into "
        "`num_grids` levels, opens/adds short size when price rises to a "
        "level, covers that unit when price falls one grid step back below "
        "it. Profits from range-bound chop or gentle downtrends; loses on "
        "sustained rallies."
    ),
    "default_params": {"range_pct": 0.30, "num_grids": 10},
}


def signals(df: pd.DataFrame, range_pct: float = 0.30, num_grids: int = 10, **_) -> pd.Series:
    close = df["close"]
    start_price = close.iloc[0]
    lower = start_price * (1 - range_pct / 2)
    upper = start_price * (1 + range_pct / 2)
    step = (upper - lower) / num_grids
    levels = [lower + i * step for i in range(num_grids + 1)]
    unit = 1.0 / num_grids

    weight = pd.Series(0.0, index=df.index)
    held_levels: set[int] = set()
    current_weight = 0.0  # negative = short exposure

    prices = close.to_numpy()
    for i, price in enumerate(prices):
        for level_idx, level_price in enumerate(levels):
            if level_idx not in held_levels and level_price <= price <= upper:
                held_levels.add(level_idx)
                current_weight = max(-1.0, current_weight - unit)
        for level_idx in list(held_levels):
            if price <= levels[level_idx] - step:
                held_levels.discard(level_idx)
                current_weight = min(0.0, current_weight + unit)
        weight.iloc[i] = current_weight
    return weight
