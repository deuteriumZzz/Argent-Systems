"""Payday anomaly: long only in a window around the mid-month payday.

Source: Quantpedia, "Payday Anomaly" (https://quantpedia.com/strategies/payday-anomaly/),
whose QC reference implementation (paperswithbacktest/awesome-systematic-trading,
static/strategies/payday-anomaly.py) buys and holds SPY on the 16th calendar
day of each month (weekend-adjusted). The underlying economic driver is
documented in Ogden, J.P. (1990), "Turn-of-Month Evaluations of Liquid
Profits and Stock Returns: A Common Explanation for the Monthly and January
Effects", Journal of Finance 45(4), 1259-1272, which shows a large fraction of
aggregate salary/interest/dividend payments cluster around standardized
mid-month and month-end payment dates, driving reinvestment-flow-related
return seasonality. Crypto has no exchange weekend closure, so this drops the
weekend adjustment and just uses a symmetric calendar-day window around the
payday. No code reused, only the documented calendar rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_payday_effect",
    "category": "seasonality",
    "source": "Quantpedia 'Payday Anomaly' / Ogden (1990) JF 45(4) standardized payment period hypothesis",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Long only when the calendar day-of-month is within `window` days of "
        "`payday` (default day 16, +/-1), flat the rest of the month."
    ),
    "default_params": {"payday": 16, "window": 1},
}


def signals(df: pd.DataFrame, payday: int = 16, window: int = 1, **_) -> pd.Series:
    day = df.index.day
    in_window = np.abs(day - payday) <= window
    return pd.Series(in_window.astype(float), index=df.index)
