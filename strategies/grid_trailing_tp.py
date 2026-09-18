"""Static grid bot with a trailing take-profit per unit (instead of a fixed
one-step target).

Source: mechanic documented by Binance's "Trailing Up / Trailing Down"
Spot & Futures Grid features, adapted here to trail the exit of each
individual grid unit rather than the whole grid's price range. See:
https://www.binance.com/en/support/faq/how-to-use-the-trailing-up-function-in-spot-grid-trading-3d987afd7906495cb4d997eccb8515bf
https://www.binance.com/en/blog/futures/leveraging-market-momentum-with-trailing-up--trailing-down-features-in-usd%E2%93%A2m-futures-grid-trading-6830518331948603129
Reimplemented from the publicly documented mechanic, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "grid_trailing_tp",
    "category": "grid",
    "source": "Binance Grid Trading Bot (Trailing Up / Trailing Down)",
    "license": "N/A",
    "description": (
        "Buys one grid unit whenever price drops to an unfilled level "
        "within [lower, upper]. Instead of selling at a fixed step above "
        "the buy price, each unit trails its own running peak since it was "
        "bought and only sells once price falls back `trail_pct` from that "
        "peak — lets winners run further in a rally instead of capping "
        "gains at one grid step."
    ),
    "default_params": {"range_pct": 0.30, "num_grids": 10, "trail_pct": 0.015},
}


def signals(
    df: pd.DataFrame,
    range_pct: float = 0.30,
    num_grids: int = 10,
    trail_pct: float = 0.015,
    **_,
) -> pd.Series:
    close = df["close"]
    start_price = close.iloc[0]
    lower = start_price * (1 - range_pct / 2)
    upper = start_price * (1 + range_pct / 2)
    step = (upper - lower) / num_grids
    levels = [lower + i * step for i in range(num_grids + 1)]
    unit = 1.0 / num_grids

    weight = pd.Series(0.0, index=df.index)
    held_levels: set[int] = set()
    peaks: dict[int, float] = {}
    current_weight = 0.0

    prices = close.to_numpy()
    for i, price in enumerate(prices):
        for level_idx, level_price in enumerate(levels):
            if level_idx not in held_levels and lower <= price <= level_price:
                held_levels.add(level_idx)
                peaks[level_idx] = price
                current_weight = min(1.0, current_weight + unit)

        for level_idx in list(held_levels):
            peaks[level_idx] = max(peaks[level_idx], price)
            if price <= peaks[level_idx] * (1 - trail_pct):
                held_levels.discard(level_idx)
                del peaks[level_idx]
                current_weight = max(0.0, current_weight - unit)

        weight.iloc[i] = current_weight
    return weight
