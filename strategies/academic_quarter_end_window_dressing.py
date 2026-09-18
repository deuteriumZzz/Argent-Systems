"""Quarter-end window dressing: long bias into calendar-quarter-end, unwind after.

Source: Lakonishok, J., Shleifer, A., Thaler, R. & Vishny, R.W. (1991),
"Window Dressing by Pension Fund Managers", American Economic Review Papers
and Proceedings 81(2), 227-231, and Ritter, J.R. (1988), "The Buying and
Selling Behavior of Individual Investors at the Turn of the Year", Journal of
Finance 43(3), 701-717 - document institutional managers systematically
buying recent winners into quarter/year-end for reporting purposes ("window
dressing"), then unwinding after the reporting date passes. Distinct from
academic_turn_of_month_seasonality.py's every-month cadence: this only fires
around the four calendar-quarter boundaries (end of Mar/Jun/Sep/Dec), which is
the specific institutional-reporting-driven window the papers study. No code
reused, only the documented calendar rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "academic_quarter_end_window_dressing",
    "category": "seasonality",
    "source": "Lakonishok, Shleifer, Thaler & Vishny (1991), 'Window Dressing by Pension Fund Managers', AER P&P 81(2) / Ritter (1988) JF 43(3)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Long the last `days_before` calendar days of each calendar quarter "
        "(Mar/Jun/Sep/Dec). Optionally shorts the first `days_after` calendar "
        "days of the new quarter (the unwind), if `allow_fade_after`. Flat "
        "otherwise."
    ),
    "default_params": {"days_before": 3, "days_after": 2, "allow_fade_after": False},
}


def signals(
    df: pd.DataFrame,
    days_before: int = 3,
    days_after: int = 2,
    allow_fade_after: bool = False,
    **_,
) -> pd.Series:
    idx = df.index
    month = idx.month
    day = idx.day
    days_in_month = idx.days_in_month

    is_quarter_end_month = (month % 3) == 0  # Mar, Jun, Sep, Dec
    is_quarter_start_month = (month % 3) == 1  # Jan, Apr, Jul, Oct

    end_window = is_quarter_end_month & (day > (days_in_month - days_before))
    after_window = is_quarter_start_month & (day <= days_after)

    weight = pd.Series(0.0, index=df.index)
    weight[end_window] = 1.0
    if allow_fade_after:
        weight[after_window] = -1.0
    return weight
