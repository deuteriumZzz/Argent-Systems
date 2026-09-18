"""Hash Ribbon miner-capitulation recovery signal.

Source: Charles Edwards (2019), "Hash Ribbons – Bitcoin's Buy Signal",
Capriole Investments. Miner capitulation (hash rate declining as
unprofitable miners shut down) historically clusters near cycle bottoms;
the signal fires on recovery, not during the capitulation itself.
Reimplemented from the published rule.
License: N/A (published on-chain metric, reimplemented from the rule).

DIFFERENT DATA CONTRACT: requires the `hash_rate` on-chain column merged
onto BTC OHLCV — see backtest/run_onchain.py. META["data_type"] = "onchain".
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "onchain_hash_ribbon",
    "category": "onchain",
    "data_type": "onchain",
    "source": "Charles Edwards (2019), Hash Ribbons, Capriole Investments",
    "license": "N/A (published on-chain metric, reimplemented from the rule)",
    "description": (
        "Tracks SMA(fast) vs SMA(slow) of hash_rate. Capitulation = fast "
        "SMA below slow SMA (miners shutting down). Buy signal fires when "
        "the fast SMA crosses back above the slow SMA (capitulation "
        "ending), held until it drops below again."
    ),
    "default_params": {"fast": 30, "slow": 60},
}


def signals(df: pd.DataFrame, fast: int = 30, slow: int = 60, **_) -> pd.Series:
    fast_sma = df["hash_rate"].rolling(fast).mean()
    slow_sma = df["hash_rate"].rolling(slow).mean()

    in_recovery = fast_sma > slow_sma
    return in_recovery.astype(float).fillna(0.0)
