"""Donchian channel breakout filtered to quiet, non-spike candles.

Source: reimplemented from iterativv/NostalgiaForInfinity buy_condition_8
(conceptually adapted, no code copied) — the original rule (labelled
"experimental, turtle-style") buys the first daily close above a 7-day
Donchian high, but only when that breakout candle's own range is small
relative to its close (a controlled push through resistance rather than a
one-candle spike/wick fakeout) and the higher-timeframe ROC isn't already
overheated. Adapted to a single timeframe by using this dataframe's own
ROC in place of the source's separate 4h check.
License: GPL-3.0 (source); this reimplementation is original code
expressing the same public rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "nfi_donchian_quiet_breakout",
    "category": "trend_following",
    "source": "reimplemented from iterativv/NostalgiaForInfinity buy_condition_8 (conceptually adapted, no code copied)",
    "license": "GPL-3.0 (source); this reimplementation is original code expressing the same public rule",
    "description": (
        "Long entry on the first bar closing above the prior "
        "`donchian_period`-bar high, gated by: candle range/close < "
        "`max_candle_range_pct` (not a spike) and ROC over `roc_period` "
        "bars < `roc_ceiling` (breakout isn't already overheated). Exits "
        "when close falls back below the Donchian mid-line."
    ),
    "default_params": {
        "donchian_period": 168,  # ~7 days on 1h bars
        "max_candle_range_pct": 0.025,
        "roc_period": 24,
        "roc_ceiling": 20.0,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    donchian_period: int = 168,
    max_candle_range_pct: float = 0.025,
    roc_period: int = 24,
    roc_ceiling: float = 20.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    dc_high = high.rolling(donchian_period).max()
    dc_low = low.rolling(donchian_period).min()
    dc_mid = (dc_high + dc_low) / 2

    prev_close = close.shift(1)
    first_break_up = (prev_close <= dc_high.shift(1)) & (close > dc_high)
    quiet_candle = ((high - low) / close) < max_candle_range_pct
    roc = close.pct_change(roc_period) * 100
    not_overheated = roc < roc_ceiling

    long_entry = first_break_up & quiet_candle & not_overheated
    long_exit = close < dc_mid

    weight = pd.Series(np.nan, index=df.index)
    # exit set first so a same-bar entry (below) can override the flatten
    weight[long_exit] = 0.0
    if allow_short:
        first_break_down = (prev_close >= dc_low.shift(1)) & (close < dc_low)
        short_entry = first_break_down & quiet_candle & (roc > -roc_ceiling)
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)
