"""Martingale grid bot — position size doubles at each deeper grid level.

Source: mechanic documented by 3Commas DCA bot's "Safety order volume scale"
/ "martingale_volume_coefficient" setting (each subsequent buy is sized as a
multiple of the previous one), applied here to a spatial price grid instead
of a time-sequenced DCA ladder. See:
https://help.3commas.io/en/articles/3108940-dca-bot-interface-and-main-settings
https://developers.3commas.io/dca-bot/create-dca-bot/
Reimplemented from the publicly documented mechanic, no code borrowed.
License: N/A (concept).

WARNING: martingale sizing means the deepest, most-adverse grid levels also
carry the largest position size — meaningfully higher risk of large
drawdowns on a sustained one-directional move than a flat/equal-size grid.
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "grid_martingale",
    "category": "grid",
    "source": (
        "3Commas DCA bot (Safety order volume scale / martingale_volume_coefficient), "
        "adapted to a spatial grid"
    ),
    "license": "N/A",
    "description": (
        "Same [lower, upper] price grid as a static grid bot, but the size "
        "bought at each level grows geometrically by `size_multiplier` the "
        "deeper (closer to `lower`) the level is — mirrors 3Commas' "
        "martingale volume-scale DCA setting. HIGHER RISK than a flat grid: "
        "a sustained move to the bottom of the range concentrates the "
        "largest buys at the worst prices."
    ),
    "default_params": {"range_pct": 0.30, "num_grids": 10, "size_multiplier": 1.3},
}


def signals(
    df: pd.DataFrame,
    range_pct: float = 0.30,
    num_grids: int = 10,
    size_multiplier: float = 1.3,
    **_,
) -> pd.Series:
    close = df["close"]
    start_price = close.iloc[0]
    lower = start_price * (1 - range_pct / 2)
    upper = start_price * (1 + range_pct / 2)
    step = (upper - lower) / num_grids
    levels = [lower + i * step for i in range(num_grids + 1)]

    # deeper level (closer to `lower`, smaller index) gets a bigger size
    raw_sizes = [size_multiplier ** (num_grids - i) for i in range(num_grids + 1)]
    total = sum(raw_sizes)
    sizes = [s / total for s in raw_sizes]

    weight = pd.Series(0.0, index=df.index)
    held_levels: set[int] = set()
    current_weight = 0.0

    prices = close.to_numpy()
    for i, price in enumerate(prices):
        for level_idx, level_price in enumerate(levels):
            if level_idx not in held_levels and lower <= price <= level_price:
                held_levels.add(level_idx)
                current_weight = min(1.0, current_weight + sizes[level_idx])
        for level_idx in list(held_levels):
            if price >= levels[level_idx] + step:
                held_levels.discard(level_idx)
                current_weight = max(0.0, current_weight - sizes[level_idx])
        weight.iloc[i] = current_weight
    return weight
