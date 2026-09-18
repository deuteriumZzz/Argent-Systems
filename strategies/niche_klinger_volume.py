"""Klinger Volume Oscillator crossover.

Source: classic concept (Stephen J. Klinger's Volume Oscillator, first
published in the 1990s; documented formula as used by StockCharts/Investopedia).
A volume-flow indicator rarely seen in crypto strategy repos compared to
price-only oscillators. Reimplemented from the publicly documented formula,
no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "niche_klinger_volume",
    "category": "trend_following",
    "source": "classic concept (Stephen Klinger's Volume Oscillator)",
    "license": "N/A",
    "description": (
        "Signed volume force (volume * trend direction, scaled by how far "
        "the daily range has traveled since the last trend flip), smoothed "
        "into a fast/slow EMA oscillator (KVO) plus its own EMA signal "
        "line. Long while KVO is above its signal line, flat/short otherwise."
    ),
    "default_params": {"fast_period": 34, "slow_period": 55, "signal_period": 13, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    fast_period: int = 34,
    slow_period: int = 55,
    signal_period: int = 13,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    hlc = (high + low + close) / 3
    trend = np.sign(hlc.diff()).replace(0, np.nan).ffill().fillna(1.0)

    dm = high - low
    trend_flip = trend != trend.shift(1)
    cm = np.zeros(len(df))
    dm_arr = dm.to_numpy()
    flip_arr = trend_flip.to_numpy()
    for i in range(len(df)):
        if i == 0 or flip_arr[i]:
            cm[i] = dm_arr[i]
        else:
            cm[i] = cm[i - 1] + dm_arr[i]
    cm_series = pd.Series(cm, index=df.index).replace(0, np.nan)

    vf = volume * (2 * (dm / cm_series - 1)).abs().fillna(0.0) * trend * 100
    kvo = vf.ewm(span=fast_period, adjust=False).mean() - vf.ewm(span=slow_period, adjust=False).mean()
    signal = kvo.ewm(span=signal_period, adjust=False).mean()

    bullish = kvo > signal
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})
