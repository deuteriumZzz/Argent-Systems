"""ADX crossing-up "trend birth" entry with directional and EMA200 filter.

Source: reimplemented from iterativv/NostalgiaForInfinity buy_condition_7
(conceptually adapted, no code copied) — the original 4h-timeframe rule
enters when ADX crosses up through 20 (a trend just becoming
statistically real, not yet mature/crowded), +DI is above -DI (bulls hold
direction), and price trades above EMA200 (long-term uptrend regime).
Adapted here to a single timeframe (no separate 4h dataframe available in
this repo's single-df interface): all three conditions are computed on the
bars we're given instead of a resampled higher timeframe.
License: GPL-3.0 (source); this reimplementation is original code
expressing the same public rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "nfi_adx_trend_birth",
    "category": "trend_following",
    "source": "reimplemented from iterativv/NostalgiaForInfinity buy_condition_7 (conceptually adapted, no code copied)",
    "license": "GPL-3.0 (source); this reimplementation is original code expressing the same public rule",
    "description": (
        "Long entry the bar ADX(period) crosses up through `adx_threshold` "
        "while +DI > -DI and close > EMA(ema_period) — catches a trend just "
        "being confirmed, not a mature/crowded one. Exits when +DI drops "
        "back below -DI or close falls back under the EMA."
    ),
    "default_params": {
        "adx_period": 14,
        "adx_threshold": 20.0,
        "ema_period": 200,
        "allow_short": False,
    },
}


def _dmi_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)
    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx


def signals(
    df: pd.DataFrame,
    adx_period: int = 14,
    adx_threshold: float = 20.0,
    ema_period: int = 200,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    plus_di, minus_di, adx = _dmi_adx(df["high"], df["low"], close, adx_period)
    ema = close.ewm(span=ema_period, adjust=False).mean()

    adx_cross_up = (adx > adx_threshold) & (adx.shift(1) <= adx_threshold)
    bulls_own_it = plus_di > minus_di
    uptrend = close > ema

    long_entry = adx_cross_up & bulls_own_it & uptrend
    long_exit = (~bulls_own_it) | (~uptrend)

    weight = pd.Series(np.nan, index=df.index)
    # exits set first so a same-bar entry (below) can override the flatten
    weight[long_exit] = 0.0
    if allow_short:
        # same ADX-crossing-up event (a trend being born); direction is
        # decided by DI/EMA instead, so a fresh trend the bears own shorts.
        short_entry = adx_cross_up & (~bulls_own_it) & (~uptrend)
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)
