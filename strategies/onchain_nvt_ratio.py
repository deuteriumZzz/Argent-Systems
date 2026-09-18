"""NVT Ratio (Network Value to Transactions) mean-reversion.

Source: Willy Woo's "NVT Ratio" (2017, Woobull research) — Bitcoin's
on-chain analogue of a P/E ratio: market cap divided by the USD value moved
on-chain. High NVT = price rich relative to actual network usage, low NVT =
price cheap relative to usage. Reimplemented from the published formula.
License: N/A (concept).

DIFFERENT DATA CONTRACT: requires on-chain columns (market_cap_usd,
transaction_volume_usd) merged onto BTC OHLCV — see backtest/run_onchain.py.
META["data_type"] = "onchain".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "onchain_nvt_ratio",
    "category": "onchain",
    "data_type": "onchain",
    "source": "Willy Woo (2017), NVT Ratio, Woobull research",
    "license": "N/A (published on-chain metric, reimplemented from the formula)",
    "description": (
        "NVT = market_cap_usd / smoothed transaction_volume_usd. Fades "
        "extremes vs. its own rolling z-score: short/reduce when NVT is "
        "unusually high (overvalued vs. usage), long when unusually low."
    ),
    "default_params": {"volume_smooth": 14, "zscore_lookback": 180, "entry_z": 1.0},
}


def signals(
    df: pd.DataFrame,
    volume_smooth: int = 14,
    zscore_lookback: int = 180,
    entry_z: float = 1.0,
    **_,
) -> pd.Series:
    smoothed_volume = df["transaction_volume_usd"].rolling(volume_smooth).mean()
    nvt = df["market_cap_usd"] / smoothed_volume.replace(0, np.nan)

    mean = nvt.rolling(zscore_lookback).mean()
    std = nvt.rolling(zscore_lookback).std()
    zscore = (nvt - mean) / std.replace(0, np.nan)

    weight = pd.Series(0.0, index=df.index)
    weight[zscore <= -entry_z] = 1.0
    weight[zscore >= entry_z] = -1.0
    return weight.fillna(0.0)
