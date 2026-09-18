"""GTAA moving-average timing rule (Faber, 2007).

Source: Faber, M.T. (2007), "A Quantitative Approach to Tactical Asset
Allocation", Journal of Wealth Management 9(4), 69-79
(https://papers.ssrn.com/sol3/papers.cfm?abstract_id=962461). GTAA's timing
rule (applied per-asset before any cross-asset allocation step, which is
what makes it usable single-asset): hold the asset when its price is above
its trailing 10-month simple moving average, otherwise move to cash. The
paper shows this single rule, applied to five very different asset
classes, cut drawdowns roughly in half with little cost to CAGR. No code
reused - just the documented moving-average-vs-price switch.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_gtaa_ma_timing",
    "category": "trend_following",
    "source": "Faber (2007), 'A Quantitative Approach to Tactical Asset Allocation', JWM 9(4)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Binary timing switch: long (weight 1.0) while close is above its "
        "trailing `ma_lookback`-bar simple moving average (Faber's 10-month "
        "SMA rule), flat (0.0) otherwise. allow_short flips the flat leg to "
        "fully short instead, for comparison against the paper's original "
        "long/cash-only design."
    ),
    "default_params": {"ma_lookback": 210, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    ma_lookback: int = 210,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    sma = close.rolling(ma_lookback, min_periods=ma_lookback).mean()

    if allow_short:
        weight = np.sign(close - sma)
    else:
        weight = (close > sma).astype(float)

    return pd.Series(weight, index=df.index).fillna(0.0).clip(-1.0, 1.0)
