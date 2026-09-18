"""Momentum crash hedge / dynamic scaling (Daniel & Moskowitz, 2016).

Source: Daniel, K., Moskowitz, T.J. (2016), "Momentum Crashes", Journal of
Financial Economics 122(2), 221-247
(https://www.nber.org/system/files/working_papers/w20439/w20439.pdf). Key
finding: momentum's worst crashes cluster in "panic states" - after the
broad market has already fallen (past cumulative return negative) while
volatility stays elevated - because a subsequent market rebound then hurts
momentum hardest. Their dynamic momentum strategy de-levers sharply in
exactly that regime. Also summarized on Quantpedia
(https://quantpedia.com/three-methods-to-fix-momentum-crashes/) as one of
three momentum-crash fixes. Reimplemented here for a single asset (no code
reused): standard trailing-return momentum, but exposure is multiplied by a
`dampen` factor whenever the market is in a detected bear+high-vol "panic"
regime.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_momentum_crash_hedge",
    "category": "trend_following",
    "source": "Daniel & Moskowitz (2016), 'Momentum Crashes', JFE 122(2)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Base signal is plain trailing-return momentum (sign of "
        "`mom_lookback`-bar return), inverse-vol sized. A 'panic state' is "
        "flagged when the trailing `bear_lookback`-bar return is negative "
        "AND realized volatility exceeds `vol_mult` times its own rolling "
        "median over the same window - the exact bear-market-plus-high-vol "
        "condition the paper ties to momentum crashes. In that regime, "
        "exposure is multiplied by `dampen` (default cuts it to 20%)."
    ),
    "default_params": {
        "mom_lookback": 126,
        "bear_lookback": 252,
        "vol_lookback": 21,
        "vol_mult": 1.5,
        "dampen": 0.2,
        "target_vol": 0.02,
        "max_leverage": 1.0,
    },
}


def signals(
    df: pd.DataFrame,
    mom_lookback: int = 126,
    bear_lookback: int = 252,
    vol_lookback: int = 21,
    vol_mult: float = 1.5,
    dampen: float = 0.2,
    target_vol: float = 0.02,
    max_leverage: float = 1.0,
    **_,
) -> pd.Series:
    close = df["close"]
    bar_return = close.pct_change()
    realized_vol = bar_return.rolling(vol_lookback).std()
    vol_median = realized_vol.rolling(bear_lookback, min_periods=vol_lookback).median()

    is_bear = close.pct_change(bear_lookback) < 0
    is_high_vol = realized_vol > (vol_mult * vol_median)
    panic = (is_bear & is_high_vol).fillna(False)

    direction = np.sign(close.pct_change(mom_lookback))
    base_size = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan).clip(upper=max_leverage)
    size = base_size.where(~panic, base_size * dampen)

    weight = (direction * size).clip(-1.0, 1.0)
    return weight.fillna(0.0)
