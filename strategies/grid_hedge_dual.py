"""Hedge grid — simultaneous long grid (below start price) and short grid
(above start price), net weight can land anywhere in [-1, 1] depending on
which side has more filled levels.

Source: mechanic documented by Pionex Futures Grid Bot's "Neutral" mode
(places buy limits below and sell/short limits above price with no initial
directional position) combined with Binance Futures "Hedge Mode" (holding
long and short positions on the same symbol simultaneously). See:
https://www.pionex.com/blog/pionex_neutral_futures_grid/
https://support.pionex.com/hc/en-us/articles/45343668185113-Futures-Grid-Bot
https://www.binance.com/en/support/faq/what-is-hedge-mode-and-how-to-use-it-360041513552
Reimplemented from the publicly documented mechanic, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "grid_hedge_dual",
    "category": "grid",
    "source": "Pionex Futures Grid Bot (Neutral mode) + Binance Futures Hedge Mode",
    "license": "N/A",
    "description": (
        "Splits [lower, upper] into `num_grids` levels around the start "
        "price with no initial position. Below the start price it behaves "
        "like a long grid (buy low, sell one step higher); above it "
        "behaves like a short grid (sell short high, cover one step "
        "lower). Net weight = long units - short units, so it can sit "
        "anywhere in [-1, 1] and profits from chop in either direction."
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
    half = num_grids // 2 or 1
    unit = 1.0 / (2 * half)

    weight = pd.Series(0.0, index=df.index)
    held_long: set[int] = set()
    held_short: set[int] = set()
    long_w = 0.0
    short_w = 0.0

    prices = close.to_numpy()
    for i, price in enumerate(prices):
        for level_idx, level_price in enumerate(levels):
            if level_price >= start_price:
                if level_idx not in held_short and level_price <= price <= upper:
                    held_short.add(level_idx)
                    short_w = min(1.0, short_w + unit)
            else:
                if level_idx not in held_long and lower <= price <= level_price:
                    held_long.add(level_idx)
                    long_w = min(1.0, long_w + unit)

        for level_idx in list(held_long):
            if price >= levels[level_idx] + step:
                held_long.discard(level_idx)
                long_w = max(0.0, long_w - unit)
        for level_idx in list(held_short):
            if price <= levels[level_idx] - step:
                held_short.discard(level_idx)
                short_w = max(0.0, short_w - unit)

        weight.iloc[i] = long_w - short_w
    return weight
