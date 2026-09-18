"""Crypto day-of-week effect: go long on whichever weekday has historically had
the best returns for this asset, learned from trailing same-weekday history.

Source: Kaiser, L. (2019), "Seasonality in the Cryptocurrency Market: New
Evidence from a Novel and Extensive Dataset", Finance Research Letters 31;
Caporale, G.M. & Plastun, A. (2019), "The Day of the Week Effect in the
Cryptocurrency Market", Finance Research Letters 31; Aharon, D.Y. & Qadan, M.
(2019), "Bitcoin and the Day-of-the-Week Effect", Finance Research Letters 31,
415-424 - all document statistically significant weekday-dependent mean
returns in Bitcoin/crypto (most consistently, an abnormally strong Monday).
Rather than hardcoding "Monday" (see academic_bitcoin_weekend_monday_effect.py
for that fixed-rule variant), this reimplements the underlying mechanic - "some
weekday has a persistently higher conditional mean return; go long only on
that weekday" - by learning the best weekday out-of-sample from a rolling
window of trailing same-weekday occurrences. No code reused, only the rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_crypto_day_of_week_effect",
    "category": "seasonality",
    "source": "Kaiser (2019) / Caporale & Plastun (2019) / Aharon & Qadan (2019) crypto day-of-week effect",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Resamples to daily closes, computes each weekday's trailing rolling "
        "mean return over its last `lookback_occurrences` occurrences "
        "(shifted by one to avoid lookahead), and goes long only on the "
        "weekday whose trailing conditional mean is currently the highest. "
        "Flat if not enough history yet."
    ),
    "default_params": {"lookback_occurrences": 52, "min_occurrences": 8, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    lookback_occurrences: int = 52,
    min_occurrences: int = 8,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    daily_close = df["close"].resample("1D").last().dropna()
    daily_ret = daily_close.pct_change()

    cond_means = {}
    for d in range(7):
        mask = daily_ret.index.dayofweek == d
        sub = daily_ret[mask]
        rolled = sub.rolling(lookback_occurrences, min_periods=min_occurrences).mean().shift(1)
        cond_means[d] = rolled.reindex(daily_ret.index).ffill()
    cond_df = pd.DataFrame(cond_means)

    have_any = cond_df.notna().any(axis=1)
    best_day = cond_df.fillna(-np.inf).idxmax(axis=1)
    today_is_best = (daily_ret.index.dayofweek == best_day) & have_any

    daily_weight = pd.Series(
        np.where(today_is_best, 1.0, (-1.0 if allow_short else 0.0)),
        index=daily_ret.index,
    )
    daily_weight[~have_any] = 0.0

    weight = daily_weight.reindex(df.index.normalize())
    weight.index = df.index
    return weight.fillna(0.0)
