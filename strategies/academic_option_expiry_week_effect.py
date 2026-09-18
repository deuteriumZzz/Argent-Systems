"""Option-expiration-week effect, adapted to crypto's monthly options calendar.

Source: Quantpedia, "Option Expiration Week Effect"
(https://quantpedia.com/strategies/option-expiration-week-effect/), whose QC
reference implementation goes long S&P 100 stocks during the week of monthly
equity-option expiration and stays in cash otherwise. Crypto options (Deribit,
CME) also expire monthly, conventionally on the last Friday of the month -
this uses only that public calendar convention (no options chain/IV data
needed) as the single-asset time-series proxy for "option-expiration week".
No code reused, only the documented calendar rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "academic_option_expiry_week_effect",
    "category": "seasonality",
    "source": "Quantpedia 'Option Expiration Week Effect', adapted to Deribit/CME's last-Friday-of-month BTC/ETH options expiry convention",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Computes the last Friday of each calendar month (the standard "
        "crypto monthly options expiry date) and goes long during the "
        "`window_days` calendar days leading up to and including it, flat "
        "otherwise."
    ),
    "default_params": {"window_days": 5},
}


def signals(df: pd.DataFrame, window_days: int = 5, **_) -> pd.Series:
    idx = df.index
    month_end = idx + pd.offsets.MonthEnd(0)
    offset_days = (month_end.dayofweek - 4) % 7  # Friday == 4
    expiry_date = (month_end - pd.to_timedelta(offset_days, unit="D")).normalize()

    days_to_expiry = (expiry_date - idx.normalize()).days
    in_window = (days_to_expiry >= 0) & (days_to_expiry < window_days)
    return pd.Series(in_window.astype(float), index=df.index)
