"""Static grid trading bot.

Source: generic mechanic shared by Binance Grid Trading, Pionex Grid Trading
Bot, 3Commas Grid bot. Reimplemented from the publicly documented mechanic,
no code borrowed from any specific bot.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "grid_trading",
    "category": "grid",
    "source": "generic concept (Binance/Pionex/3Commas grid bots)",
    "license": "N/A",
    "description": (
        "Splits [lower, upper] into `num_grids` equally spaced price levels "
        "around the start price. Buys one grid unit when price drops to a "
        "level, sells it when price rises one grid step above that level."
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
    current_weight = 0.0

    prices = close.to_numpy()
    for i, price in enumerate(prices):
        for level_idx, level_price in enumerate(levels):
            if level_idx not in held_levels and lower <= price <= level_price:
                held_levels.add(level_idx)
                current_weight = min(1.0, current_weight + unit)
        for level_idx in list(held_levels):
            if price >= levels[level_idx] + step:
                held_levels.discard(level_idx)
                current_weight = max(0.0, current_weight - unit)
        weight.iloc[i] = current_weight
    return weight
