"""Ease of Movement (EOM) threshold strategy.

Source: classic concept (Richard W. Arms Jr.'s Ease of Movement, 1960s;
documented formula as used by StockCharts/Investopedia). EOM relates the
size of a bar's price move to the volume required to produce it — a big
price move on light volume means price is moving "easily", a small move on
heavy volume means it is meeting resistance. Reimplemented from the
publicly documented formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "volume_ease_of_movement",
    "category": "trend_following",
    "source": "classic concept (Richard Arms' Ease of Movement)",
    "license": "N/A",
    "description": (
        "Distance Moved = midpoint((high+low)/2) change bar-to-bar. Box "
        "Ratio = (volume / `volume_scale`) / (high-low). Raw EOM = Distance "
        "Moved / Box Ratio, smoothed by an `sma_period`-bar SMA. Long while "
        "smoothed EOM > `entry_threshold` (price rising easily on light "
        "volume), flat (or short if `allow_short`) while below "
        "-`entry_threshold`, holds previous weight in between."
    ),
    "default_params": {
        "sma_period": 14,
        "volume_scale": 1_000_000.0,
        "entry_threshold": 0.0,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    sma_period: int = 14,
    volume_scale: float = 1_000_000.0,
    entry_threshold: float = 0.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, volume = df["high"], df["low"], df["volume"]

    midpoint = (high + low) / 2
    distance_moved = midpoint.diff()
    box_ratio = (volume / volume_scale) / (high - low).replace(0, np.nan)
    raw_eom = (distance_moved / box_ratio).fillna(0.0)
    eom = raw_eom.rolling(sma_period).mean()

    weight = pd.Series(np.nan, index=df.index)
    weight[eom > entry_threshold] = 1.0
    weight[eom < -entry_threshold] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
