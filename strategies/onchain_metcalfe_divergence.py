"""Price / active-address divergence (accumulation-distribution proxy).

Source: on-chain "divergence" heuristic discussed in Metcalfe's-Law-style
Bitcoin valuation research (e.g. Woobull's NVAM/network-value-to-Metcalfe
commentary) — when active addresses keep growing while price stalls or
falls, that's read as quiet accumulation (usage outpacing price); the
mirror (price up, addresses flat/down) is read as distribution into
retail euphoria. Reimplemented from the published observation.
License: N/A (published on-chain research, reimplemented from the observation).

DIFFERENT DATA CONTRACT: requires the `unique_addresses` on-chain column
merged onto BTC OHLCV — see backtest/run_onchain.py.
META["data_type"] = "onchain".
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "onchain_metcalfe_divergence",
    "category": "onchain",
    "data_type": "onchain",
    "source": "Metcalfe's-Law-style divergence heuristic (Woobull NVAM commentary)",
    "license": "N/A (published on-chain research, reimplemented from the observation)",
    "description": (
        "Compares trailing `lookback`-day growth of price vs. smoothed "
        "unique_addresses. Long when addresses are growing faster than "
        "price (accumulation), short/flat when price is growing faster "
        "than addresses (distribution)."
    ),
    "default_params": {"smooth": 14, "lookback": 30},
}


def signals(df: pd.DataFrame, smooth: int = 14, lookback: int = 30, **_) -> pd.Series:
    price_growth = df["close"].pct_change(lookback)
    address_growth = df["unique_addresses"].rolling(smooth).mean().pct_change(lookback)

    divergence = address_growth - price_growth
    return divergence.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).fillna(0.0)
