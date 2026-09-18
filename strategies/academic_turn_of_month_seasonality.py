"""Turn-of-the-month seasonality (Ariel, 1987).

Source: Ariel, R.A. (1987), "A Monthly Effect in Stock Returns", Journal of
Financial Economics 18(1), 161-174. Also a strategy page on
paperswithbacktest.com / Quantpedia
(https://quantpedia.com/strategies/turn-of-the-month-in-equity-indexes/),
which buys an equity index a day or two before month-end and sells a few
trading days into the new month. Reimplemented here using calendar days
(crypto has no exchange trading-day calendar to align to) - no code reused,
only the documented calendar window.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "academic_turn_of_month_seasonality",
    "category": "trend_following",
    "source": "Ariel (1987), 'A Monthly Effect in Stock Returns', JFE 18(1) / Quantpedia 'Turn of the Month in Equity Indexes'",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Long only during the calendar turn-of-month window: the last "
        "`days_before` calendar days of a month through the first "
        "`days_after` calendar days of the next month. Flat the rest of "
        "the month."
    ),
    "default_params": {"days_before": 1, "days_after": 3},
}


def signals(
    df: pd.DataFrame,
    days_before: int = 1,
    days_after: int = 3,
    **_,
) -> pd.Series:
    day = df.index.day
    days_in_month = df.index.days_in_month

    in_window = (day > (days_in_month - days_before)) | (day <= days_after)
    return pd.Series(in_window.astype(float), index=df.index)
