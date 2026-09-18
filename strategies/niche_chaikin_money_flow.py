"""Chaikin Money Flow (CMF) threshold strategy.

Source: classic concept (Marc Chaikin's Chaikin Money Flow, 1980s;
documented formula as used by StockCharts/Investopedia). A volume-weighted
accumulation/distribution oscillator, distinct from the pure price-action
indicators covered elsewhere in this repo. Reimplemented from the publicly
documented formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "niche_chaikin_money_flow",
    "category": "trend_following",
    "source": "classic concept (Marc Chaikin's Chaikin Money Flow)",
    "license": "N/A",
    "description": (
        "Money Flow Multiplier = ((close-low)-(high-close))/(high-low), "
        "times volume, summed over `period` bars and normalized by summed "
        "volume. Long while CMF > `entry_threshold`, flat (or short if "
        "`allow_short`) while CMF < -`entry_threshold`, holds previous "
        "weight in between."
    ),
    "default_params": {"period": 20, "entry_threshold": 0.05, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    period: int = 20,
    entry_threshold: float = 0.05,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    rng = (high - low).replace(0, float("nan"))
    mfm = ((close - low) - (high - close)) / rng
    mfv = (mfm * volume).fillna(0.0)

    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, float("nan"))

    weight = pd.Series(float("nan"), index=df.index)
    weight[cmf > entry_threshold] = 1.0
    weight[cmf < -entry_threshold] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
