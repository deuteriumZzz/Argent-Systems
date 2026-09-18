"""Shooting Star candlestick reversal.

Source: je-suis-tm/quant-trading, "Shooting Star backtest.py"
(https://github.com/je-suis-tm/quant-trading/blob/master/Shooting%20Star%20backtest.py) -
Apache License 2.0. The source's 8 conditions include two that peek at the
*next* bar's high/close to "confirm" the pattern (`shift(-1)`) - that's
look-ahead bias, so it's dropped here; only the 6 conditions knowable as of
the current bar are kept. It also sizes "small body" against the mean body
of the *entire* series (again look-ahead, and it forgets to take the
absolute value before averaging) - replaced with a trailing rolling mean of
absolute body size. The exit rule (fixed stop/target % or a max holding
period) is kept. No code copied, only the documented candle-shape rule.

This pattern is a pure reversal/short signal with no long side at all, so
unlike this repo's other pattern modules `allow_short` defaults to True -
leaving it False would make the strategy permanently flat.
License: Apache-2.0 (source repo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "pattern_shooting_star",
    "category": "mean_reversion",
    "source": "je-suis-tm/quant-trading (Shooting Star backtest.py)",
    "license": "Apache-2.0",
    "description": (
        "Flags a Shooting Star: red candle, little/no lower wick, small "
        "body (vs its trailing average), an upper wick at least 2x the "
        "body, following two bars of rising closes. Goes short on the "
        "trigger, exits after `stop_pct` adverse-or-favourable move or "
        "`holding_period` bars, whichever first. Flat (never short) if "
        "`allow_short` is False."
    ),
    "default_params": {
        "lower_wick_max": 0.2,
        "body_size_mult": 0.5,
        "body_lookback": 20,
        "stop_pct": 0.05,
        "holding_period": 7,
        "allow_short": True,
    },
}


def signals(
    df: pd.DataFrame,
    lower_wick_max: float = 0.2,
    body_size_mult: float = 0.5,
    body_lookback: int = 20,
    stop_pct: float = 0.05,
    holding_period: int = 7,
    allow_short: bool = True,
    **_,
) -> pd.Series:
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

    body = (open_ - close).abs()
    avg_body = body.rolling(body_lookback).mean()

    red = open_ >= close
    small_lower_wick = (close - low) < lower_wick_max * (close - open_).abs()
    small_body = body < body_size_mult * avg_body
    long_upper_wick = (high - open_) >= 2 * (open_ - close)
    uptrend = (close >= close.shift(1)) & (close.shift(1) >= close.shift(2))

    trigger = (red & small_lower_wick & small_body & long_upper_wick & uptrend).fillna(False).to_numpy()
    close_np = close.to_numpy()
    n = len(df)

    weight = np.zeros(n)
    short_weight = -1.0 if allow_short else 0.0
    in_trade = False
    entry_price = 0.0
    bars_held = 0

    for i in range(n):
        if in_trade:
            bars_held += 1
            if abs(close_np[i] / entry_price - 1) > stop_pct or bars_held >= holding_period:
                in_trade = False
        elif trigger[i]:
            in_trade = True
            entry_price = close_np[i]
            bars_held = 0
        weight[i] = short_weight if in_trade else 0.0

    return pd.Series(weight, index=df.index)
