"""Always-on perpetual funding-rate carry (cash-and-carry basis trade).

Source: standard crypto market-neutral desk trade — long spot + short the
perpetual future, funded by whichever side pays. Documented widely (e.g.
Deribit Insights, Kaiko/Amberdata basis-trade writeups); reimplemented from
the public mechanic, no code borrowed.
License: N/A (concept).

DIFFERENT DATA CONTRACT: signals() here expects `df` to have a
`funding_rate` column (from data/funding.py), not OHLCV. META["data_type"]
= "funding_rate" tells backtest/run.py's regular price-based runner to skip
these — use backtest/run_carry.py instead. weight > 0 means "on" the
long-spot/short-perp trade, which earns +funding_rate * weight per funding
interval when funding_rate > 0 (and pays when it's negative).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "carry_static_positive_funding",
    "category": "carry",
    "data_type": "funding_rate",
    "source": "generic concept (crypto perp cash-and-carry basis trade)",
    "license": "N/A",
    "description": (
        "Always fully allocated to the long-spot/short-perp basis trade, "
        "regardless of the current funding rate. The dumb baseline every "
        "other carry strategy here must beat."
    ),
    "default_params": {},
}


def signals(df: pd.DataFrame, **_) -> pd.Series:
    return pd.Series(1.0, index=df.index)
