"""Awesome Oscillator zero-line cross + "saucer" early-entry pattern.

Source: je-suis-tm/quant-trading, "Awesome Oscillator backtest.py"
(https://github.com/je-suis-tm/quant-trading/blob/master/Awesome%20Oscillator%20backtest.py) -
Apache License 2.0. The source combines a base zero-line-cross position
(AO above/below zero) with an early "saucer" trigger - two candles of one
color followed by an opposite-color candle while AO is still on the wrong
side of zero but curling back toward it - stacked via a signal/cumsum
scheme with a stop on repeated entries. Reimplemented here as a single
continuous position: the zero-line cross sets the base stance, and the
saucer pattern can flip that stance a bar or two earlier. No code copied,
only the documented AO formula (SMA5 - SMA34 of the H/L midpoint) and
saucer trigger conditions.
License: Apache-2.0 (source repo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "pattern_awesome_oscillator",
    "category": "trend_following",
    "source": "je-suis-tm/quant-trading (Awesome Oscillator backtest.py)",
    "license": "Apache-2.0",
    "description": (
        "AO = SMA(fast) - SMA(slow) of the (high+low)/2 midpoint. Base "
        "stance is long while AO > 0, short (or flat if `allow_short` is "
        "False) while AO < 0. The 'saucer' pattern - two same-colour "
        "candles followed by an opposite-colour one while AO is curling "
        "back toward zero from below/above - flips the stance a bar or two "
        "ahead of the actual zero-line cross."
    ),
    "default_params": {"fast_period": 5, "slow_period": 34, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    fast_period: int = 5,
    slow_period: int = 34,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    midpoint = (df["high"] + df["low"]) / 2
    ao = midpoint.rolling(fast_period).mean() - midpoint.rolling(slow_period).mean()

    open_, close = df["open"], df["close"]
    red = open_ > close
    green = open_ < close

    bullish_saucer = (
        red
        & green.shift(1)
        & green.shift(2)
        & (ao.shift(1) > ao.shift(2))
        & (ao.shift(1) < 0)
        & (ao < 0)
    )
    bearish_saucer = (
        green
        & red.shift(1)
        & red.shift(2)
        & (ao.shift(1) < ao.shift(2))
        & (ao.shift(1) > 0)
        & (ao > 0)
    )

    short_weight = -1.0 if allow_short else 0.0
    weight = pd.Series(
        np.where(ao > 0, 1.0, np.where(ao < 0, short_weight, np.nan)), index=df.index
    )
    weight[bullish_saucer] = 1.0
    weight[bearish_saucer] = short_weight
    return weight.ffill().fillna(0.0)
