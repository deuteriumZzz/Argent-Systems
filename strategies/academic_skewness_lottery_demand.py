"""Skewness / lottery-demand anomaly (MAX effect).

Source: Bali, T.G., Cakici, N., Whitelaw, R.F. (2011), "Maxing Out: Stocks
as Lotteries and the Cross-Section of Expected Returns", Journal of
Financial Economics 99(2), 427-446. Related cross-sectional test on
paperswithbacktest.com / Quantpedia
(https://quantpedia.com/strategies/skewness-effect-in-commodities/), which
longs the lowest-skew quintile and shorts the highest-skew quintile of a
futures universe. Reimplemented here for a single asset (no code reused):
periods of unusually high recent positive skewness (lottery-like return
distribution, in demand and thus overpriced per the paper) are faded; low
or negative skew periods are bought.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_skewness_lottery_demand",
    "category": "mean_reversion",
    "source": "Bali, Cakici & Whitelaw (2011), 'Maxing Out', JFE 99(2) / Quantpedia 'Skewness Effect in Commodities'",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Rolling skewness of bar returns over `skew_window`, standardized "
        "against its own trailing mean/std over `norm_window`. Weight is "
        "-tanh(sensitivity * skew_zscore): high positive skew (lottery "
        "demand) -> short/flat, low or negative skew -> long. With "
        "allow_short=False the negative side is clipped to 0."
    ),
    "default_params": {
        "skew_window": 20,
        "norm_window": 60,
        "sensitivity": 1.0,
        "allow_short": True,
    },
}


def signals(
    df: pd.DataFrame,
    skew_window: int = 20,
    norm_window: int = 60,
    sensitivity: float = 1.0,
    allow_short: bool = True,
    **_,
) -> pd.Series:
    bar_return = df["close"].pct_change()
    skew = bar_return.rolling(skew_window).skew()

    skew_mean = skew.rolling(norm_window).mean()
    skew_std = skew.rolling(norm_window).std()
    skew_z = ((skew - skew_mean) / skew_std).replace([np.inf, -np.inf], np.nan)

    weight = -np.tanh(sensitivity * skew_z)
    weight = weight.fillna(0.0)
    if not allow_short:
        weight = weight.clip(lower=0.0)
    return weight
