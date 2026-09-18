"""Bollinger/Keltner "squeeze" coil-and-release breakout.

Source: reimplemented from iterativv/NostalgiaForInfinity buy_condition_166
(conceptually adapted, no code copied), itself an adaptation of the
publicly documented "Squeeze Momentum" indicator (LazyBear). Volatility is
"squeezed" whenever the Bollinger Bands sit fully inside the Keltner
Channel — price is coiling. This strategy buys the bar the squeeze first
releases (BB moves back outside KC) after having been coiled for a
meaningful stretch, and the release breaks upward through the Bollinger
upper band.
License: GPL-3.0 (source, NFI's adaptation); this reimplementation is
original code expressing the same publicly documented indicator concept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "nfi_squeeze_momentum_release",
    "category": "volatility_breakout",
    "source": "reimplemented from iterativv/NostalgiaForInfinity buy_condition_166 (conceptually adapted, no code copied); underlying concept is the publicly documented Squeeze Momentum indicator (LazyBear)",
    "license": "GPL-3.0 (source, NFI's adaptation); this reimplementation is original code expressing the same publicly documented indicator concept",
    "description": (
        "Squeeze is 'on' when the Bollinger Bands (period, num_std) sit "
        "entirely inside the Keltner Channel (period, atr_mult). Long "
        "entry on the bar the squeeze turns off after being on for at "
        "least `min_squeeze_bars` of the prior `lookback_bars`, provided "
        "close breaks above the Bollinger upper band that same bar. Exits "
        "when close falls back below the Bollinger midline."
    ),
    "default_params": {
        "bb_period": 20,
        "bb_std": 2.0,
        "kc_period": 20,
        "kc_atr_mult": 1.5,
        "lookback_bars": 24,
        "min_squeeze_bars": 12,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_atr_mult: float = 1.5,
    lookback_bars: int = 24,
    min_squeeze_bars: int = 12,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    bb_mid = close.rolling(bb_period).mean()
    bb_stdev = close.rolling(bb_period).std()
    bb_upper = bb_mid + bb_std * bb_stdev
    bb_lower = bb_mid - bb_std * bb_stdev

    kc_mid = close.rolling(kc_period).mean()
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(kc_period).mean()
    kc_upper = kc_mid + kc_atr_mult * atr
    kc_lower = kc_mid - kc_atr_mult * atr

    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_count = squeeze_on.rolling(lookback_bars).sum()

    squeeze_released = (~squeeze_on) & squeeze_on.shift(1)
    coiled_enough = squeeze_count.shift(1) >= min_squeeze_bars
    breaks_up = close > bb_upper

    long_entry = squeeze_released & coiled_enough & breaks_up
    long_exit = close < bb_mid

    weight = pd.Series(np.nan, index=df.index)
    # exit set first so a same-bar entry (below) can override the flatten
    weight[long_exit] = 0.0
    if allow_short:
        breaks_down = close < bb_lower
        short_entry = squeeze_released & coiled_enough & breaks_down
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)
