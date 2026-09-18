"""Classic Dollar-Cost Averaging (DCA).

Source: generic public-domain concept (same mechanic as Binance "Auto-Invest",
Coinbase recurring buy). Used here mainly as a dumb baseline every other
strategy must beat.
License: N/A (concept, no code borrowed).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "classic_dca",
    "category": "accumulation",
    "source": "generic concept (Binance Auto-Invest / Coinbase recurring buy)",
    "license": "N/A",
    "description": (
        "Buys a fixed fraction of the target allocation every `interval_bars` "
        "bars, ramping exposure from 0% to 100% and holding after that."
    ),
    "default_params": {"interval_bars": 24 * 7, "num_buys": 20},
}


def signals(df: pd.DataFrame, interval_bars: int = 24 * 7, num_buys: int = 20, **_) -> pd.Series:
    weight = pd.Series(0.0, index=df.index)
    step = 1.0 / num_buys
    for i, bar_idx in enumerate(range(0, len(df), interval_bars)):
        if i >= num_buys:
            break
        weight.iloc[bar_idx:] = min(1.0, (i + 1) * step)
    return weight
