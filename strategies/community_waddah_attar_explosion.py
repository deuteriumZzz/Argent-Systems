"""Waddah Attar Explosion (WAE) — momentum vs. volatility gate.

Source: Pine Script community indicator, originally created by forex
trader Ahmad Waddah Attar (~2007) and reimplemented under many names
across TradingView, MT4/MT5, ProRealTime and cTrader community libraries
(see e.g. tradingview.com/script/nDYHOjah, luxalgo.com/library/indicator/
waddah-attar-explosion, prorealcode.com/prorealtime-indicators/waddah-attar
-explosion). Ported to Python from the publicly documented formula: a MACD
acceleration histogram ("trend power") is only trusted when it clears both
a Bollinger-Band-width "explosion line" (volatility is actually expanding)
and a rolling-ATR "dead zone" (filters out noise in dead/ranging markets).
Source: Pine Script community indicator, ported to Python from the
publicly documented formula.
License: N/A (public indicator formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "community_waddah_attar_explosion",
    "category": "volatility_breakout",
    "source": "Pine Script community indicator (Waddah Attar Explosion, Ahmad Waddah Attar)",
    "license": "N/A (public indicator formula)",
    "description": (
        "Trend = (MACD(fast,slow) - MACD.shift(1)) * sensitivity. Explosion "
        "line = Bollinger Band width (bb_period, bb_std). Dead zone = "
        "rolling mean true range over dz_period * dz_mult. Long while Trend "
        "is positive and exceeds both the explosion line and the dead zone "
        "(strong bullish momentum backed by real volatility expansion). "
        "Short (if `allow_short`) on the symmetric bearish read. Flat "
        "whenever momentum is weak, squeezed, or inside the dead zone."
    ),
    "default_params": {
        "macd_fast": 20,
        "macd_slow": 40,
        "sensitivity": 150.0,
        "bb_period": 20,
        "bb_std": 2.0,
        "dz_period": 100,
        "dz_mult": 3.7,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    macd_fast: int = 20,
    macd_slow: int = 40,
    sensitivity: float = 150.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    dz_period: int = 100,
    dz_mult: float = 3.7,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    macd = close.ewm(span=macd_fast, adjust=False).mean() - close.ewm(span=macd_slow, adjust=False).mean()
    trend = (macd - macd.shift(1)) * sensitivity

    bb_mid = close.rolling(bb_period).mean()
    bb_stdev = close.rolling(bb_period).std()
    explosion_line = 2 * bb_std * bb_stdev
    _ = bb_mid  # midline unused directly, kept for readability of the BB construction

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    dead_zone = tr.rolling(dz_period).mean() * dz_mult

    bullish_strong = (trend > 0) & (trend > explosion_line) & (trend > dead_zone)
    bearish_strong = (trend < 0) & (trend.abs() > explosion_line) & (trend.abs() > dead_zone)
    weak = ~bullish_strong & ~bearish_strong

    weight = pd.Series(np.nan, index=df.index)
    weight[weak] = 0.0
    weight[bearish_strong] = -1.0 if allow_short else 0.0
    weight[bullish_strong] = 1.0
    return weight.ffill().fillna(0.0)
