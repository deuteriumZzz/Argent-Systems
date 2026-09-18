"""Rolling VWAP trend-following filter (price above/below fair value).

Source: classic concept (Volume Weighted Average Price, standard exchange
execution benchmark; documented formula as used by StockCharts/Investopedia).
`reversion_vwap.py` elsewhere in this repo *fades* price back toward VWAP
on the assumption a stretch away from fair value mean-reverts. This file
does the opposite: it treats a sustained close above a rising rolling VWAP
as confirmation that buyers are willing to pay above the volume-weighted
average, i.e. a trend-following regime filter, not a reversion trade.
Reimplemented from the publicly documented formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "volume_vwap_trend_filter",
    "category": "trend_following",
    "source": "classic concept (Volume Weighted Average Price)",
    "license": "N/A",
    "description": (
        "Rolling VWAP over `vwap_period` bars from typical price * volume. "
        "Long while close is above VWAP AND VWAP itself has risen over the "
        "last `slope_lookback` bars (price paying above a rising fair "
        "value = trend confirmed). Flat (or short if `allow_short`) while "
        "close is below a falling VWAP. Holds previous weight otherwise."
    ),
    "default_params": {"vwap_period": 20, "slope_lookback": 5, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    vwap_period: int = 20,
    slope_lookback: int = 5,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close, volume = df["close"], df["volume"]
    typical = (df["high"] + df["low"] + close) / 3
    pv = (typical * volume).rolling(vwap_period).sum()
    vol = volume.rolling(vwap_period).sum()
    vwap = pv / vol.replace(0, np.nan)

    vwap_rising = vwap > vwap.shift(slope_lookback)
    above = (close > vwap) & vwap_rising
    below = (close < vwap) & ~vwap_rising

    weight = pd.Series(np.nan, index=df.index)
    weight[above] = 1.0
    weight[below] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
