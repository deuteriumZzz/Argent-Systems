"""Value averaging — a documented alternative to DCA that targets a growing
*portfolio value* curve, buying more on dips and trimming into rallies to
stay on the target path (instead of DCA's fixed buy size regardless of
price).

Source: Michael E. Edleson, "Value Averaging: The Safe and Easy Strategy
for Higher Investment Returns" (1988 paper / 1993 book) — classic,
widely-cited alternative to dollar-cost averaging in the investing
literature. This file uses the simplest linear target-value path from that
family (ramping to full allocation over `num_periods` checkpoints); Edleson
also describes geometric growth paths, not implemented here.
Reimplemented from the publicly described method, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "dca_value_averaging",
    "category": "accumulation",
    "source": "classic concept (Michael Edleson's Value Averaging, 1988/1993)",
    "license": "N/A",
    "description": (
        "At every checkpoint (every `interval_bars` bars, for `num_periods` "
        "checkpoints), computes a target portfolio value that ramps "
        "linearly up to 100% of capital, then buys or sells just enough to "
        "bring the current mark-to-market value onto that target path — "
        "buying more after a drop, trimming after a rally. Unlike classic "
        "DCA it can reduce exposure, not just add to it."
    ),
    "default_params": {"interval_bars": 24 * 7, "num_periods": 20},
}


def signals(df: pd.DataFrame, interval_bars: int = 24 * 7, num_periods: int = 20, **_) -> pd.Series:
    prices = df["close"].to_numpy()
    n = len(prices)
    weight = pd.Series(0.0, index=df.index)

    qty = 0.0  # position size, in units of (fraction of capital) / price
    checkpoint = 0
    for i in range(n):
        price = prices[i]
        if i % interval_bars == 0 and checkpoint < num_periods:
            target_value = min(1.0, (checkpoint + 1) / num_periods)
            current_value = qty * price
            qty = max(0.0, qty + (target_value - current_value) / price)
            checkpoint += 1
        weight.iloc[i] = min(1.0, qty * price)
    return weight
