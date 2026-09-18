"""Intermediate-horizon ("echo") momentum (Novy-Marx, 2012).

Source: Novy-Marx, R. (2012), "Is Momentum Really Momentum?", Journal of
Financial Economics 103(3), 429-453
(https://rnm.simon.rochester.edu/research/IMRM.pdf). Finding: the return
from 12 to 7 months before formation (the "intermediate" horizon) predicts
future returns better than the more recent 6-to-2-month return that most
momentum strategies emphasize - performance is driven by "what happened a
year ago", not "what happened last quarter". Reimplemented here for a
single asset (no code reused): the position signal is built from that
older window instead of the standard recent-return window.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_intermediate_horizon_momentum",
    "category": "trend_following",
    "source": "Novy-Marx (2012), 'Is Momentum Really Momentum?', JFE 103(3)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Computes the 'echo' return between `far_lookback` and "
        "`near_lookback` bars ago (e.g. 12-to-7-months-ago), which the "
        "paper shows is the part of trailing momentum with real predictive "
        "power. Direction follows the sign of that older-window return, "
        "sized to a target volatility using trailing realized vol - "
        "deliberately ignores the most recent months entirely, unlike "
        "standard trailing-return momentum."
    ),
    "default_params": {
        "far_lookback": 252,
        "near_lookback": 147,
        "vol_lookback": 20,
        "target_vol": 0.02,
        "max_leverage": 1.0,
    },
}


def signals(
    df: pd.DataFrame,
    far_lookback: int = 252,
    near_lookback: int = 147,
    vol_lookback: int = 20,
    target_vol: float = 0.02,
    max_leverage: float = 1.0,
    **_,
) -> pd.Series:
    close = df["close"]
    echo_return = close.shift(near_lookback) / close.shift(far_lookback) - 1.0
    bar_return = close.pct_change()
    realized_vol = bar_return.rolling(vol_lookback).std()

    direction = np.sign(echo_return)
    size = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan).clip(upper=max_leverage)

    weight = (direction * size).clip(-1.0, 1.0)
    return weight.fillna(0.0)
