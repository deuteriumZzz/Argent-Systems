"""Volume-Weighted MACD (VW-MACD) crossover.

Source: classic concept (Volume-Weighted MACD, a documented MACD variant
that substitutes Volume Weighted Moving Averages for the usual EMAs —
formula as described by StockCharts/Investopedia's MACD family articles
and various charting-platform indicator libraries). Standard MACD in this
repo (`macd_trend.py`) uses plain EMAs of price; this variant weights each
bar's contribution to the moving average by its volume, so heavy-volume
bars move the line more than thin ones. Reimplemented from the publicly
documented formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "volume_weighted_macd",
    "category": "trend_following",
    "source": "classic concept (Volume-Weighted MACD)",
    "license": "N/A",
    "description": (
        "VWMA(n) = rolling_sum(close*volume, n) / rolling_sum(volume, n). "
        "VW-MACD = VWMA(fast_period) - VWMA(slow_period), signal line = "
        "EMA(VW-MACD, signal_period). Long while VW-MACD > signal, flat "
        "(or short if `allow_short`) while below."
    ),
    "default_params": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9,
        "allow_short": False,
    },
}


def _vwma(close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    return (close * volume).rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def signals(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close, volume = df["close"], df["volume"]

    vw_macd = _vwma(close, volume, fast_period) - _vwma(close, volume, slow_period)
    signal_line = vw_macd.ewm(span=signal_period, adjust=False).mean()

    bullish = vw_macd > signal_line
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})
