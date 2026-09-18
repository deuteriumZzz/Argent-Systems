"""Funding-rate carry, sized by how rich funding is versus its own history.

Source: standard basis-trade refinement — size the position proportionally
to a rolling z-score of the funding rate, so capital leans harder into
unusually generous funding regimes instead of a flat on/off. Reimplemented
from the public mechanic.
License: N/A (concept).

Same funding_rate data contract as carry_static_positive_funding.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "carry_funding_zscore",
    "category": "carry",
    "data_type": "funding_rate",
    "source": "generic concept (z-score-sized crypto basis trade)",
    "license": "N/A",
    "description": (
        "Weight = clip(zscore(funding_rate, lookback), 0, cap) / cap: only "
        "takes the long-spot/short-perp side, sized up when funding is "
        "unusually high relative to its own recent history."
    ),
    "default_params": {"lookback": 90, "cap": 2.0},
}


def signals(df: pd.DataFrame, lookback: int = 90, cap: float = 2.0, **_) -> pd.Series:
    funding = df["funding_rate"]
    mean = funding.rolling(lookback).mean()
    std = funding.rolling(lookback).std()
    zscore = (funding - mean) / std.replace(0, np.nan)

    weight = (zscore.clip(lower=0, upper=cap) / cap).fillna(0.0)
    return weight
