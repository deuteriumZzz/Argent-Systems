"""Coinbase Premium extreme fade (sentiment-exhaustion reversal).

Source: market commentary observing that Coinbase premium spikes to
unusually extreme levels (in either direction) often coincide with
short-term sentiment exhaustion — a euphoric premium blow-off tends to
precede a pullback, a deep discount panic tends to precede a bounce
(documented informally in CryptoQuant/on-chain-analyst commentary around
the Coinbase Premium Index). Reimplemented from the public observation.
License: N/A (published market indicator, reimplemented from the observation).

Same cross_exchange data contract as arbitrage_coinbase_premium_trend.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "arbitrage_premium_extreme_reversal",
    "category": "cross_exchange",
    "data_type": "cross_exchange",
    "source": "Coinbase Premium extreme-reversal observation (CryptoQuant/on-chain commentary)",
    "license": "N/A (published market indicator, reimplemented from the observation)",
    "description": (
        "Fades the Coinbase premium's own rolling z-score extremes: short "
        "BTC when the premium spikes unusually high (euphoria), long when "
        "it plunges unusually low (panic discount)."
    ),
    "default_params": {"lookback": 30, "entry_z": 1.5},
}


def signals(df: pd.DataFrame, lookback: int = 30, entry_z: float = 1.5, **_) -> pd.Series:
    premium = df["coinbase_premium_pct"]
    mean = premium.rolling(lookback).mean()
    std = premium.rolling(lookback).std()
    zscore = (premium - mean) / std.replace(0, np.nan)

    weight = pd.Series(0.0, index=df.index)
    weight[zscore >= entry_z] = -1.0
    weight[zscore <= -entry_z] = 1.0
    return weight.fillna(0.0)
