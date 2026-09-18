"""Multi-level DCA with widening step size between safety orders.

Source: 3Commas DCA bot "Price Deviation" + "Step Multiplier" mechanic
(https://help.3commas.io/en/articles/11983699-dca-bot-averaging-order-settings-explained) —
each subsequent safety order triggers further away from the previous fill,
deviation_i = initial_deviation * step_multiplier^(i-1). Volume is kept
constant across orders here (that's the martingale/volume-scale variant,
see grid_martingale.py); this file only widens the *spacing*.
Reimplemented from the publicly documented mechanic, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "dca_widening_step",
    "category": "accumulation",
    "source": "3Commas DCA bot (Price Deviation / Step Multiplier)",
    "license": "N/A",
    "description": (
        "Places a base order immediately, then up to `num_safety_orders` "
        "equal-size buys, each triggered when price drops below the last "
        "fill by a deviation that grows by `step_multiplier` per level. "
        "Exits the whole position when price rises `take_profit_pct` above "
        "the volume-weighted average entry."
    ),
    "default_params": {
        "initial_deviation_pct": 0.02,
        "step_multiplier": 1.5,
        "num_safety_orders": 8,
        "take_profit_pct": 0.03,
    },
}


def signals(
    df: pd.DataFrame,
    initial_deviation_pct: float = 0.02,
    step_multiplier: float = 1.5,
    num_safety_orders: int = 8,
    take_profit_pct: float = 0.03,
    **_,
) -> pd.Series:
    prices = df["close"].to_numpy()
    weight = pd.Series(0.0, index=df.index)
    unit = 1.0 / (num_safety_orders + 1)

    position = 0.0
    orders_filled = 0
    last_fill_price = None
    avg_price = 0.0

    for i, price in enumerate(prices):
        if orders_filled == 0:
            avg_price = price
            position = unit
            orders_filled = 1
            last_fill_price = price
        elif orders_filled <= num_safety_orders:
            deviation = initial_deviation_pct * (step_multiplier ** (orders_filled - 1))
            trigger_price = last_fill_price * (1 - deviation)
            if price <= trigger_price:
                avg_price = (avg_price * position + price * unit) / (position + unit)
                position = min(1.0, position + unit)
                orders_filled += 1
                last_fill_price = price

        if position > 0 and price >= avg_price * (1 + take_profit_pct):
            position = 0.0
            orders_filled = 0
            last_fill_price = None
            avg_price = 0.0

        weight.iloc[i] = position
    return weight


def demo() -> None:
    import numpy as np

    n = 300
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.full(n, -0.001)))  # steady grind down
    df = pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0},
        index=idx,
    )
    w = signals(df, **META["default_params"])
    assert w.between(-1.0, 1.0).all()
    assert w.iloc[0] > 0, "base order should fire on bar 0"
    assert w.iloc[-1] > w.iloc[0], "steady downtrend should trigger extra safety orders"
    print("dca_widening_step demo OK")


if __name__ == "__main__":
    demo()
