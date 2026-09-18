"""Overnight vs. intraday session return anomaly (adapted for 24/7 crypto).

Source: Cliff, M.T., Cooper, M.J., Gulen, H. (2008), "Return Differences
between Trading and Non-Trading Hours: Like Night and Day", working paper;
Lou, D., Polk, C., Skouras, S. (2019), "A Tug of War: Overnight versus
Intraday Expected Returns", Journal of Financial Economics 134(1), 192-213.
Also a strategy page on paperswithbacktest.com / Quantpedia
(https://quantpedia.com/strategies/market-sentiment-and-an-overnight-anomaly/).
The original anomaly is defined by a stock exchange's close/open; crypto
trades 24/7, so this reimplements the underlying mechanic - "returns are not
evenly distributed across the day, and the historically stronger half-day
session should be favored" - using a fixed UTC hour split as the crypto
analogue of the close/open boundary. No code reused, only the rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_overnight_session_anomaly",
    "category": "mean_reversion",
    "source": "Cliff, Cooper & Gulen (2008) / Lou, Polk & Skouras (2019) overnight-return anomaly, adapted to a fixed UTC session split for 24/7 markets",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Splits each day at `split_hour` UTC into an 'A' session "
        "([0, split_hour)) and a 'B' session ([split_hour, 24)). Using only "
        "past bars (shifted by one), tracks each session's rolling mean "
        "bar-return and goes long during whichever session currently has "
        "the higher historical mean return; requires intraday bars "
        "(hourly or finer) to have any effect."
    ),
    "default_params": {"split_hour": 8, "lookback": 168, "min_periods": 24, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    split_hour: int = 8,
    lookback: int = 168,
    min_periods: int = 24,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    in_session_a = pd.Series(df.index.hour < split_hour, index=df.index)
    bar_return = df["close"].pct_change()

    ret_a = bar_return.where(in_session_a)
    ret_b = bar_return.where(~in_session_a)

    mean_a = ret_a.rolling(lookback, min_periods=min_periods).mean().shift(1)
    mean_b = ret_b.rolling(lookback, min_periods=min_periods).mean().shift(1)

    session_a_better = mean_a > mean_b
    long_now = np.where(in_session_a, session_a_better, ~session_a_better)

    short_weight = -1.0 if allow_short else 0.0
    weight = pd.Series(np.where(long_now, 1.0, short_weight), index=df.index)
    weight[mean_a.isna() | mean_b.isna()] = 0.0
    return weight
