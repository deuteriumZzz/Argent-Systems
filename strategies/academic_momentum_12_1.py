"""12-1 (skip-month) momentum (Jegadeesh & Titman, 1993).

Source: Jegadeesh, N., Titman, S. (1993), "Returns to Buying Winners and
Selling Losers: Implications for Stock Market Efficiency", Journal of
Finance 48(1), 65-91. This is the canonical momentum formation window used
across the momentum literature (also underlying Asness, Moskowitz &
Pedersen (2013), "Value and Momentum Everywhere", JF 68(3)) and mirrored on
Quantpedia (https://quantpedia.com/strategies/momentum-factor-effect-in-stocks/).
Distinct from academic_time_series_momentum.py: that file uses the full
trailing-return window with no skip; this one specifically skips the most
recent month to strip out short-term reversal, per the paper's own finding
that including the last month hurts the signal.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_momentum_12_1",
    "category": "trend_following",
    "source": "Jegadeesh & Titman (1993), 'Returns to Buying Winners and Selling Losers', JF 48(1)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Classic 12-1 momentum: measures return from `lookback` bars ago to "
        "`skip` bars ago, deliberately excluding the most recent `skip` "
        "bars (avoids the well-documented short-term reversal in the last "
        "month). Position direction follows the sign of that skip-month "
        "return, sized to a target volatility using trailing realized vol."
    ),
    "default_params": {
        "lookback": 252,
        "skip": 21,
        "vol_lookback": 20,
        "target_vol": 0.02,
        "max_leverage": 1.0,
    },
}


def signals(
    df: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    vol_lookback: int = 20,
    target_vol: float = 0.02,
    max_leverage: float = 1.0,
    **_,
) -> pd.Series:
    close = df["close"]
    skip_month_return = close.shift(skip) / close.shift(lookback) - 1.0
    bar_return = close.pct_change()
    realized_vol = bar_return.rolling(vol_lookback).std()

    direction = np.sign(skip_month_return)
    size = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan).clip(upper=max_leverage)

    weight = (direction * size).clip(-1.0, 1.0)
    return weight.fillna(0.0)
