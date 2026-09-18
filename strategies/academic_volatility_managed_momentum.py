"""Volatility-managed momentum (Barroso & Santa-Clara, 2015).

Source: Barroso, P., Santa-Clara, P. (2015), "Momentum Has Its Moments",
Journal of Financial Economics 116(1), 111-120. Also a Quantpedia entry
(https://quantpedia.com/strategies/momentum-factor-effect-in-stocks/,
"three methods to fix momentum crashes" discussion at
https://quantpedia.com/three-methods-to-fix-momentum-crashes/). The paper's
core result: scaling a momentum position by the inverse of its own trailing
realized VARIANCE (not volatility/std) roughly halves crash risk and nearly
doubles the Sharpe ratio versus a constant-notional momentum position.
Reimplemented here for a single asset (no code reused): momentum direction
from a plain trailing return, sized by inverse EWMA realized variance
(the paper's FRV estimator uses daily squared returns over the trailing six
months) rather than the simple rolling std used elsewhere in this repo.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_volatility_managed_momentum",
    "category": "trend_following",
    "source": "Barroso & Santa-Clara (2015), 'Momentum Has Its Moments', JFE 116(1)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Momentum direction from the sign of the trailing `lookback`-bar "
        "return. Position size is the paper's constant-risk scaling: "
        "target_vol^2 / EWMA(realized daily variance over `var_lookback` "
        "bars) - inverse VARIANCE weighting (not inverse std), which reacts "
        "faster to vol spikes than the plain-vol scaling used in "
        "academic_time_series_momentum.py, directly targeting Barroso & "
        "Santa-Clara's crash-mitigation mechanism."
    ),
    "default_params": {
        "lookback": 252,
        "var_lookback": 126,
        "target_vol": 0.02,
        "max_leverage": 1.0,
    },
}


def signals(
    df: pd.DataFrame,
    lookback: int = 252,
    var_lookback: int = 126,
    target_vol: float = 0.02,
    max_leverage: float = 1.0,
    **_,
) -> pd.Series:
    close = df["close"]
    trailing_return = close.pct_change(lookback)
    bar_return = close.pct_change()

    # EWMA of squared daily returns = RiskMetrics-style realized variance,
    # matching the paper's emphasis on recent-weighted realized variance.
    ewma_variance = (bar_return**2).ewm(span=var_lookback, min_periods=var_lookback).mean()

    direction = np.sign(trailing_return)
    size = (
        (target_vol**2 / ewma_variance)
        .replace([np.inf, -np.inf], np.nan)
        .clip(upper=max_leverage)
    )

    weight = (direction * size).clip(-1.0, 1.0)
    return weight.fillna(0.0)
