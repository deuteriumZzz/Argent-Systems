"""Funding-rate carry driven by the persistence of funding itself.

Source: documented observation that perp funding is autocorrelated over
short windows (a positive print tends to be followed by more positive
prints during a trending market, and vice versa) — widely discussed in
crypto-derivatives research writeups (e.g. Kaiko/Amberdata/Deribit Insights
basis-trade notes). Reimplemented from the public observation.
License: N/A (concept).

Same funding_rate data contract as carry_static_positive_funding.py.
Unlike the other carry_* strategies here, this one can also go short the
basis trade (i.e. short spot / long perp) when funding has been
persistently negative, since that side is just as directly investable.
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "carry_funding_momentum",
    "category": "carry",
    "data_type": "funding_rate",
    "source": "generic concept (funding-rate autocorrelation / persistence)",
    "license": "N/A",
    "description": (
        "Weight = sign of the trailing `lookback`-period average funding "
        "rate: long-spot/short-perp while recent funding has averaged "
        "positive, the mirror trade while it has averaged negative."
    ),
    "default_params": {"lookback": 21},
}


def signals(df: pd.DataFrame, lookback: int = 21, **_) -> pd.Series:
    trailing_mean = df["funding_rate"].rolling(lookback).mean()
    return trailing_mean.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).fillna(0.0)
