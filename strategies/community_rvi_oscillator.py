"""Relative Vigor Index (RVI) signal-line crossover.

Source: classic concept (John Ehlers' Relative Vigor Index, published in
Stocks & Commodities magazine, 2002; documented formula as used by
StockCharts/Investopedia and reproduced widely as a TradingView community
indicator). Measures the "vigor" of a trend by comparing where a bar
closes relative to where it opened against its overall high-low range,
using a symmetric 4-bar weighted filter on both numerator and denominator
to smooth out noise. Reimplemented from the publicly documented formula,
no code borrowed.
License: N/A (public indicator formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "community_rvi_oscillator",
    "category": "trend_following",
    "source": "classic concept (John Ehlers' Relative Vigor Index)",
    "license": "N/A (public indicator formula)",
    "description": (
        "Numerator/denominator are each a symmetric 4-bar weighted filter "
        "([1,2,2,1]/6) of (close-open) and (high-low) respectively, summed "
        "over `period` bars via an SMA of each filtered series, then "
        "divided: RVI = SMA(num,period)/SMA(denom,period). Signal = the "
        "same [1,2,2,1]/6 filter applied to RVI itself. Long while RVI > "
        "signal, short (if `allow_short`) while below."
    ),
    "default_params": {"period": 10, "allow_short": False},
}


def _symmetric_filter(series: pd.Series) -> pd.Series:
    return (
        series + 2 * series.shift(1) + 2 * series.shift(2) + series.shift(3)
    ) / 6


def signals(
    df: pd.DataFrame,
    period: int = 10,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    open_, high, low, close = df["open"], df["high"], df["low"], df["close"]

    num = _symmetric_filter(close - open_).rolling(period).sum()
    denom = _symmetric_filter(high - low).rolling(period).sum()
    rvi = num / denom.replace(0, np.nan)
    signal_line = _symmetric_filter(rvi)

    bullish = rvi > signal_line
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})
