"""Active-address growth momentum ("Metcalfe's Law" style).

Source: widely-cited on-chain research applying Metcalfe's Law to Bitcoin
(e.g. Alabi 2017, "Digital blockchain networks appear to be following
Metcalfe's Law"; Woobull/Coin Metrics network-value-vs-addresses research)
— network usage growth (active addresses) tends to lead or confirm price
trend. Reimplemented from the published observation.
License: N/A (published on-chain research, reimplemented from the observation).

DIFFERENT DATA CONTRACT: requires the `unique_addresses` on-chain column
merged onto BTC OHLCV — see backtest/run_onchain.py.
META["data_type"] = "onchain".
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "onchain_active_address_momentum",
    "category": "onchain",
    "data_type": "onchain",
    "source": "Metcalfe's-Law-style on-chain research (Alabi 2017; Woobull/Coin Metrics)",
    "license": "N/A (published on-chain research, reimplemented from the observation)",
    "description": (
        "Weight = sign of the trailing `lookback`-day growth rate in "
        "smoothed unique_addresses: long while network usage is expanding, "
        "flat/short while it's contracting."
    ),
    "default_params": {"smooth": 14, "lookback": 30},
}


def signals(df: pd.DataFrame, smooth: int = 14, lookback: int = 30, **_) -> pd.Series:
    smoothed = df["unique_addresses"].rolling(smooth).mean()
    growth = smoothed.pct_change(lookback)
    return growth.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).fillna(0.0)
