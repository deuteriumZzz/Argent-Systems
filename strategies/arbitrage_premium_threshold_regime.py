"""Coinbase Premium discrete regime switch (on/off, not continuous).

Source: same "Coinbase Premium Index" concept as
arbitrage_coinbase_premium_trend.py, but treated as a discrete regime gate
with hysteresis instead of a continuous sign-of-premium follower — only
commits once the premium clears a meaningful threshold, avoiding whipsaws
on noise around zero.
License: N/A (published market indicator, reimplemented from the concept).

Same cross_exchange data contract as arbitrage_coinbase_premium_trend.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "arbitrage_premium_threshold_regime",
    "category": "cross_exchange",
    "data_type": "cross_exchange",
    "source": "Coinbase Premium Index concept (CryptoQuant/Kaiko market commentary)",
    "license": "N/A (published market indicator, reimplemented from the concept)",
    "description": (
        "Long BTC once coinbase_premium_pct rises above `entry_threshold`, "
        "flips short once it drops below `-entry_threshold`, otherwise "
        "holds the last regime (hysteresis around zero)."
    ),
    "default_params": {"entry_threshold": 0.1},
}


def signals(df: pd.DataFrame, entry_threshold: float = 0.1, **_) -> pd.Series:
    premium = df["coinbase_premium_pct"]

    weight = pd.Series(np.nan, index=df.index)
    weight[premium >= entry_threshold] = 1.0
    weight[premium <= -entry_threshold] = -1.0
    return weight.ffill().fillna(0.0)
