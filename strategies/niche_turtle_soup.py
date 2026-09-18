"""Turtle Soup - fade failed Donchian breakouts (Linda Bradford Raschke).

Source: classic concept, published as "Turtle Soup" in Linda Raschke &
Laurence Connors' book "Street Smarts" (1996) - a documented counter-trend
fade of failed N-bar breakouts of the classic Turtle Trading channel
(already implemented as a follower in `donchian_channel.py`; this is the
mirror-image fade). Reimplemented from the publicly documented rule, no
code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "niche_turtle_soup",
    "category": "mean_reversion",
    "source": "classic concept (Linda Raschke / Laurence Connors, 'Street Smarts' - Turtle Soup)",
    "license": "N/A",
    "description": (
        "If the low breaks below the prior `period`-bar low but closes back "
        "above it (a failed breakdown), goes long for `hold_bars` bars. "
        "Mirror case (failed breakout above the prior high) goes short for "
        "`hold_bars` bars if `allow_short`, else stays flat."
    ),
    "default_params": {"period": 20, "hold_bars": 5, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    period: int = 20,
    hold_bars: int = 5,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    prior_low = low.shift(1).rolling(period).min()
    prior_high = high.shift(1).rolling(period).max()

    false_breakdown = (low < prior_low) & (close > prior_low)
    false_breakout = (high > prior_high) & (close < prior_high)

    weight = [0.0] * len(df)
    remaining = 0
    current = 0.0
    fb_down = false_breakdown.to_numpy()
    fb_up = false_breakout.to_numpy()

    for i in range(len(df)):
        if remaining > 0:
            weight[i] = current
            remaining -= 1
            if remaining == 0:
                current = 0.0
        elif fb_down[i]:
            current = 1.0
            weight[i] = current
            remaining = hold_bars - 1
        elif fb_up[i] and allow_short:
            current = -1.0
            weight[i] = current
            remaining = hold_bars - 1
        else:
            weight[i] = 0.0

    return pd.Series(weight, index=df.index)
