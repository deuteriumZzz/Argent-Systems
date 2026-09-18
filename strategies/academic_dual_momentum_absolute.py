"""Dual momentum - absolute momentum leg (Antonacci, 2014).

Source: Antonacci, G. (2014), "Dual Momentum Investing: An Innovative
Strategy for Higher Returns with Lower Risk", McGraw-Hill (the GEM /
"Global Equities Momentum" strategy). Also summarized by Quantpedia
(https://quantpedia.com/active-dual-momentum-gtaa-strategy/). Dual momentum
combines a *relative* momentum leg (pick the best of several assets - not
implementable single-asset) with an *absolute* momentum leg (compare one
asset's own trailing return to a risk-free/cash return; only be invested
when it clears that bar). This file reimplements only the absolute leg,
the single-asset-implementable half of the strategy - no code reused.
Distinct from academic_time_series_momentum.py: this is a binary in/cash
switch with no shorting and no vol-target sizing, matching Antonacci's
exact rule ("be long only when trailing return beats T-bills, otherwise
move to the safe asset").
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_dual_momentum_absolute",
    "category": "trend_following",
    "source": "Antonacci (2014), 'Dual Momentum Investing', McGraw-Hill (GEM absolute momentum leg)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Computes the trailing `lookback`-bar return and compares it to "
        "`risk_free_rate` (per-window, default 0). Long-only by default: "
        "weight is 1.0 when trailing return beats the risk-free hurdle, "
        "else 0.0 (moves to 'cash'). With allow_short=True the switch goes "
        "fully short instead of flat below the hurdle, for comparison."
    ),
    "default_params": {
        "lookback": 252,
        "risk_free_rate": 0.0,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    lookback: int = 252,
    risk_free_rate: float = 0.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    trailing_return = close.pct_change(lookback)
    excess = trailing_return - risk_free_rate

    if allow_short:
        weight = np.sign(excess)
    else:
        weight = (excess > 0).astype(float)

    return pd.Series(weight, index=df.index).fillna(0.0).clip(-1.0, 1.0)
