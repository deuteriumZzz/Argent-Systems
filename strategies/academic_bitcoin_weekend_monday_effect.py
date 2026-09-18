"""Fixed-rule Bitcoin weekday effect: long on the documented strong day(s),
flat/short on the documented weak day(s) - no learning, hardcoded calendar rule.

Source: Aharon, D.Y. & Qadan, M. (2019), "Bitcoin and the Day-of-the-Week
Effect", Finance Research Letters 31, 415-424, which finds Bitcoin's strongest
and most statistically significant abnormal mean return falls on Monday, with
weekend (Saturday/Sunday) sessions comparatively weaker/more volatile.
Distinct from academic_crypto_day_of_week_effect.py (which learns the best
weekday adaptively from rolling history): this reimplements the paper's
literal finding as a fixed calendar rule. No code reused, only the rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_bitcoin_weekend_monday_effect",
    "category": "seasonality",
    "source": "Aharon & Qadan (2019), 'Bitcoin and the Day-of-the-Week Effect', Finance Research Letters 31",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Fixed calendar rule: long on `long_days` (default Monday, dayofweek "
        "0), short on `short_days` (default Saturday/Sunday, 5 and 6), flat "
        "the remaining weekdays."
    ),
    "default_params": {"long_days": [0], "short_days": [5, 6]},
}


def signals(
    df: pd.DataFrame,
    long_days: list[int] | None = None,
    short_days: list[int] | None = None,
    **_,
) -> pd.Series:
    long_days = long_days if long_days is not None else [0]
    short_days = short_days if short_days is not None else [5, 6]

    dow = df.index.dayofweek
    weight = pd.Series(0.0, index=df.index)
    weight[np.isin(dow, long_days)] = 1.0
    weight[np.isin(dow, short_days)] = -1.0
    return weight
