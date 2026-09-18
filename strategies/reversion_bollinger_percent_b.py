"""Bollinger %B mean reversion (fades extremes back to the mean).

Unlike `bollinger_breakout.py` (which buys strength on a breakout above the
upper band), this trades the opposite read of the same bands: %B = (close -
lower) / (upper - lower) measures where price sits inside the channel, and
this strategy fades %B back toward the midline once it pokes outside [0, 1].

Source: classic technical-analysis concept (John Bollinger's %B, published
alongside Bollinger Bands). Reimplemented from the published formula, no
code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "reversion_bollinger_percent_b",
    "category": "mean_reversion",
    "source": "classic concept (John Bollinger's %B)",
    "license": "N/A",
    "description": (
        "Goes long when %B drops below `oversold` (close at/below the lower "
        "band), exits when %B recovers back above `exit_level` (near the "
        "midline). Optionally shorts when %B rises above `overbought`, "
        "covering the same way from the top."
    ),
    "default_params": {
        "period": 20,
        "num_std": 2.0,
        "oversold": 0.05,
        "overbought": 0.95,
        "exit_level": 0.5,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    oversold: float = 0.05,
    overbought: float = 0.95,
    exit_level: float = 0.5,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    percent_b = (close - lower) / (upper - lower).replace(0, np.nan)

    long_entries = percent_b < oversold
    short_entries = percent_b > overbought
    # flatten on the bar %B crosses the exit level from either side
    crossed_up = (percent_b >= exit_level) & (percent_b.shift(1) < exit_level)
    crossed_down = (percent_b <= exit_level) & (percent_b.shift(1) > exit_level)
    flat = crossed_up | crossed_down

    weight = pd.Series(np.nan, index=df.index)
    weight[flat] = 0.0
    weight[long_entries] = 1.0
    weight[short_entries] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
