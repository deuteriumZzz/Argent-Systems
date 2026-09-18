"""Trend-quality-scaled momentum (Baltas & Kosowski, 2013).

Source: Baltas, A.N., Kosowski, R. (2013), "Momentum Strategies in Futures
Markets and Trend-Following Funds", published version in the Journal of
Banking & Finance and listed as a strategy on paperswithbacktest.com
(https://paperswithbacktest.com/strategies/momentum-strategies-in-futures-markets-and-trend-following-funds).
The paper shows time-series momentum forecasts improve once position sizing
accounts for how "clean" the trend actually is, not just its sign and
volatility - a trend built from one or two large jumps is less reliable
than a steady drift, even with the same net return. Reimplemented here for
a single asset (no code reused): trend "quality" is measured with the R^2
of a rolling linear regression of log-price on time (a standard trend-
pervasiveness proxy), used purely as a conviction multiplier on top of
ordinary vol-targeted trailing-return momentum.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_regression_r2_trend_strength",
    "category": "trend_following",
    "source": "Baltas & Kosowski (2013), 'Momentum Strategies in Futures Markets and Trend-Following Funds'",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Base signal: sign of `mom_lookback`-bar trailing return, inverse-"
        "vol sized. Multiplied by the R^2 of a rolling linear regression "
        "of log(close) on the bar index over `mom_lookback` bars - a "
        "steady, low-noise trend scores near 1 and gets full size; a "
        "choppy trend that happens to have the same net sign scores near 0 "
        "and gets scaled down, even though direction is unchanged."
    ),
    "default_params": {
        "mom_lookback": 63,
        "vol_lookback": 20,
        "target_vol": 0.02,
        "max_leverage": 1.0,
    },
}


def _rolling_r2(log_close: np.ndarray) -> float:
    n = len(log_close)
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = log_close.mean()
    sxx = ((x - x_mean) ** 2).sum()
    if sxx == 0:
        return 0.0
    sxy = ((x - x_mean) * (log_close - y_mean)).sum()
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    fitted = intercept + slope * x
    ss_res = ((log_close - fitted) ** 2).sum()
    ss_tot = ((log_close - y_mean) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def signals(
    df: pd.DataFrame,
    mom_lookback: int = 63,
    vol_lookback: int = 20,
    target_vol: float = 0.02,
    max_leverage: float = 1.0,
    **_,
) -> pd.Series:
    close = df["close"]
    direction = np.sign(close.pct_change(mom_lookback))

    bar_return = close.pct_change()
    realized_vol = bar_return.rolling(vol_lookback).std()
    base_size = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan).clip(upper=max_leverage)

    log_close = np.log(close)
    trend_quality = log_close.rolling(mom_lookback).apply(_rolling_r2, raw=True)

    weight = (direction * base_size * trend_quality).clip(-1.0, 1.0)
    return weight.fillna(0.0)
