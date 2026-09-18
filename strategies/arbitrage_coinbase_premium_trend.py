"""Coinbase Premium trend-following (US institutional flow proxy).

Source: "Coinbase Premium Index" — a widely tracked crypto market
indicator (published by CryptoQuant, discussed in Kaiko/Glassnode market
commentary) treating Coinbase's price premium over Binance as a proxy for
US institutional buying pressure (Coinbase skews institutional/US retail,
Binance skews global/Asia retail). Reimplemented from the public concept.
License: N/A (published market indicator, reimplemented from the concept).

DIFFERENT DATA CONTRACT: requires the `coinbase_premium_pct` column merged
onto BTC close — see data/cross_exchange.py / backtest/run_cross_exchange.py.
META["data_type"] = "cross_exchange". This is NOT literal arbitrage
execution — see that module's docstring for why.
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "arbitrage_coinbase_premium_trend",
    "category": "cross_exchange",
    "data_type": "cross_exchange",
    "source": "Coinbase Premium Index concept (CryptoQuant/Kaiko market commentary)",
    "license": "N/A (published market indicator, reimplemented from the concept)",
    "description": (
        "Long BTC while the smoothed Coinbase premium is positive (net US "
        "buying pressure), flat/short while it's negative (net selling / "
        "discount)."
    ),
    "default_params": {"smooth": 3},
}


def signals(df: pd.DataFrame, smooth: int = 3, **_) -> pd.Series:
    premium = df["coinbase_premium_pct"].rolling(smooth).mean()
    return premium.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).fillna(0.0)
