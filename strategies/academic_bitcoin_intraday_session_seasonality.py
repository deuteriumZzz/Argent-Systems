"""Intraday seasonality in Bitcoin: hold only during a fixed high-return UTC window.

Source: Quantpedia, "Overnight Seasonality in Bitcoin"
(https://quantpedia.com/strategies/intraday-seasonality-in-bitcoin/); QuantConnect
reference implementation at paperswithbacktest/awesome-systematic-trading,
static/strategies/intraday-seasonality-in-bitcoin.py, which opens a BTC long at
22:00 UTC and closes it two hours later at 00:00 UTC, based on documented
intraday return seasonality in Bitcoin (elevated returns during the US-evening
/ Asia-pre-open UTC window). Reimplemented here as a generic UTC time-of-day
window (no code reused, only the documented rule) so the window is a tunable
parameter rather than hardcoded.
License: N/A (published academic/quant strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "academic_bitcoin_intraday_session_seasonality",
    "category": "seasonality",
    "source": "Quantpedia 'Overnight Seasonality in Bitcoin' / paperswithbacktest awesome-systematic-trading static/strategies/intraday-seasonality-in-bitcoin.py",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Long only during a fixed UTC hour window (default 22:00-00:00, the "
        "documented high-return Bitcoin session), flat the rest of the day. "
        "Requires intraday (hourly or finer) bars to have any effect."
    ),
    "default_params": {"start_hour": 22, "window_hours": 2},
}


def signals(df: pd.DataFrame, start_hour: int = 22, window_hours: int = 2, **_) -> pd.Series:
    hour = df.index.hour
    end_hour = (start_hour + window_hours) % 24
    if start_hour < end_hour:
        in_window = (hour >= start_hour) & (hour < end_hour)
    else:
        # window wraps past midnight
        in_window = (hour >= start_hour) | (hour < end_hour)
    return pd.Series(in_window.astype(float), index=df.index)
