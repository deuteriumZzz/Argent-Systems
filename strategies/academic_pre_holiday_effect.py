"""Pre-holiday effect, adapted as a recurring low-participation calendar window.

Source: Ariel, R.A. (1990), "High Stock Returns before Holidays: Existence and
Evidence on Possible Causes", Journal of Finance 45(5), 1611-1626 - finds
abnormally high mean returns on the trading day(s) immediately preceding
exchange holidays. Crypto has no exchange closures, so this uses the same US
federal holiday calendar (via pandas' built-in USFederalHolidayCalendar - the
recurring dates traditional markets treat as holidays) purely as a proxy for
the recurring low-institutional-participation windows the original effect is
attributed to, and goes long the `days_before` calendar day(s) preceding each
one. No code reused, only the documented rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

META = {
    "name": "academic_pre_holiday_effect",
    "category": "seasonality",
    "source": "Ariel (1990), 'High Stock Returns before Holidays', Journal of Finance 45(5), proxied via the US federal holiday calendar",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Goes long on the `days_before` calendar day(s) immediately "
        "preceding each US federal holiday (a recurring low-liquidity-window "
        "proxy, since crypto itself never closes), flat otherwise."
    ),
    "default_params": {"days_before": 1},
}


def signals(df: pd.DataFrame, days_before: int = 1, **_) -> pd.Series:
    idx = df.index
    naive_dates = (idx.tz_convert(None) if idx.tz is not None else idx).normalize()

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(
        start=naive_dates.min() - pd.Timedelta(days=14),
        end=naive_dates.max() + pd.Timedelta(days=1),
    )

    target_dates = set()
    for h in holidays:
        for k in range(1, days_before + 1):
            target_dates.add((h - pd.Timedelta(days=k)).normalize())

    in_window = naive_dates.isin(target_dates)
    return pd.Series(in_window.astype(float), index=df.index)
