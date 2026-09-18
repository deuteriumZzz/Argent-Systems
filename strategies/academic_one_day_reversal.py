"""Finest-horizon (single-bar) short-term reversal.

Source: Lo, A.W. & MacKinlay, A.C. (1990), "When Are Contrarian Profits Due to
Stock Market Overreaction?", Review of Financial Studies 3(2), 175-205, and
Jegadeesh & Titman (1995), "Short-Horizon Return Reversals and the Bid-Ask
Spread", Journal of Financial Intermediation 4(2) - both document that the
finest available return horizon (next trading day / next print) shows the
strongest contrarian autocorrelation of all short-term reversal horizons,
distinct from the ~1-week horizon in academic_short_term_reversal.py and the
multi-week horizon in academic_long_term_overreaction_reversal.py. Adapted
here to crypto's continuous 24/7 bars: the horizon is simply the single most
recent bar. No code reused, only the documented rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_one_day_reversal",
    "category": "mean_reversion",
    "source": "Lo & MacKinlay (1990) RFS 3(2) / Jegadeesh & Titman (1995) JFI 4(2) finest-horizon return reversal",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Fades the single most recent bar's return, normalized by a short "
        "trailing volatility window and squashed through tanh."
    ),
    "default_params": {"lookback": 1, "vol_lookback": 24, "sensitivity": 4.0},
}


def signals(
    df: pd.DataFrame,
    lookback: int = 1,
    vol_lookback: int = 24,
    sensitivity: float = 4.0,
    **_,
) -> pd.Series:
    close = df["close"]
    trailing_return = close.pct_change(lookback)
    vol = close.pct_change().rolling(vol_lookback).std() * np.sqrt(lookback)

    normalized = (-trailing_return / vol).replace([np.inf, -np.inf], np.nan)
    weight = np.tanh(normalized * sensitivity)
    return weight.fillna(0.0)
