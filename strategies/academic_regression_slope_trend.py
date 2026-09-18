"""Regression slope t-stat trend signal (Levine & Pedersen, 2016).

Source: Levine, A., Pedersen, L.H. (2016), "Which Trend Is Your Friend?",
Financial Analysts Journal 72(3), 51-66
(https://www.pm-research.com/content/iijfinanajourn/72/3/51). The paper
compares the three common ways to build a trend signal - sign of trailing
return, moving-average crossover, and a rolling linear-regression trend -
and finds them highly correlated but the regression-based measure the
statistically cleanest, since its t-statistic captures both direction and
significance in one number. Reimplemented here for a single asset (no code
reused): rolling OLS of log(close) on the bar index, using the slope's
t-statistic directly as the (bounded) position signal.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_regression_slope_trend",
    "category": "trend_following",
    "source": "Levine & Pedersen (2016), 'Which Trend Is Your Friend?', Financial Analysts Journal 72(3)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Fits a rolling linear regression of log(close) on the bar index "
        "over `lookback` bars and takes the slope's t-statistic as the "
        "signal - positive and significant slope means a strong, reliable "
        "uptrend. The t-stat is clipped to +/-`tstat_cap` and rescaled to "
        "[-1, 1], so full position size requires both the right sign and "
        "statistical significance, not just a positive slope."
    ),
    "default_params": {"lookback": 63, "tstat_cap": 3.0},
}


def _slope_tstat(y: np.ndarray) -> float:
    n = len(y)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = ((x - x_mean) ** 2).sum()
    if sxx == 0 or n <= 2:
        return 0.0
    sxy = ((x - x_mean) * (y - y_mean)).sum()
    slope = sxy / sxx
    residuals = y - (y_mean + slope * (x - x_mean))
    dof = n - 2
    s2 = (residuals**2).sum() / dof
    se = np.sqrt(s2 / sxx) if sxx > 0 else np.nan
    if not se or se == 0 or np.isnan(se):
        return 0.0
    return slope / se


def signals(
    df: pd.DataFrame,
    lookback: int = 63,
    tstat_cap: float = 3.0,
    **_,
) -> pd.Series:
    log_close = np.log(df["close"])
    tstat = log_close.rolling(lookback).apply(_slope_tstat, raw=True)

    weight = (tstat / tstat_cap).clip(-1.0, 1.0)
    return weight.fillna(0.0)
