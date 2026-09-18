"""Puell Multiple miner-profitability regime.

Source: David Puell (2019), "The Bitcoin Puell Multiple", popularized via
LookIntoBitcoin. Daily USD value issued to miners divided by its own
365-day moving average — low values historically coincide with miner
profitability squeezes and cycle bottoms, high values with blow-off tops.
Reimplemented from the published formula.
License: N/A (published on-chain metric, reimplemented from the formula).

DIFFERENT DATA CONTRACT: requires the `miners_revenue_usd` on-chain column
merged onto BTC OHLCV — see backtest/run_onchain.py.
META["data_type"] = "onchain".
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "onchain_puell_multiple",
    "category": "onchain",
    "data_type": "onchain",
    "source": "David Puell (2019), Puell Multiple, LookIntoBitcoin",
    "license": "N/A (published on-chain metric, reimplemented from the formula)",
    "description": (
        "Puell Multiple = miners_revenue_usd / SMA(365) of "
        "miners_revenue_usd. Long below `low_band` (miner-profitability "
        "trough, historically an accumulation zone), flat/short above "
        "`high_band` (blow-off-top zone)."
    ),
    "default_params": {"ma_window": 365, "low_band": 0.5, "high_band": 4.0},
}


def signals(
    df: pd.DataFrame, ma_window: int = 365, low_band: float = 0.5, high_band: float = 4.0, **_
) -> pd.Series:
    revenue = df["miners_revenue_usd"]
    puell = revenue / revenue.rolling(ma_window).mean()

    weight = pd.Series(0.0, index=df.index)
    weight[puell <= low_band] = 1.0
    weight[puell >= high_band] = -1.0
    return weight.fillna(0.0)
