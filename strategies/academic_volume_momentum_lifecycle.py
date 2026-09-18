"""Volume-conditioned momentum life-cycle (Lee & Swaminathan, 2000).

Source: Lee, C.M.C., Swaminathan, B. (2000), "Price Momentum and Trading
Volume", Journal of Finance 55(5), 2017-2069
(https://www.jstor.org/stable/222397). Key finding: past trading volume
predicts the momentum life-cycle - winners/losers that arrived on
unusually heavy volume behave like "late-stage" momentum and revert
faster, while momentum built on low/normal volume is more likely to
persist. Reimplemented here for a single asset (no code reused): a
standard momentum signal is damped whenever recent volume is elevated
relative to its own trailing average, per the paper's volume-conditioned
persistence result.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_volume_momentum_lifecycle",
    "category": "trend_following",
    "source": "Lee & Swaminathan (2000), 'Price Momentum and Trading Volume', JF 55(5)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Base signal: sign of `mom_lookback`-bar trailing return, inverse-"
        "vol sized. Volume is z-scored against its trailing "
        "`vol_avg_lookback`-bar mean/std; when that z-score exceeds "
        "`high_volume_z` (unusually heavy volume - the paper's marker of "
        "late-stage, reversal-prone momentum), position size is damped by "
        "1 / (1 + excess z-score), floored at `min_damp`."
    ),
    "default_params": {
        "mom_lookback": 126,
        "vol_avg_lookback": 63,
        "high_volume_z": 1.0,
        "min_damp": 0.2,
        "vol_lookback": 20,
        "target_vol": 0.02,
        "max_leverage": 1.0,
    },
}


def signals(
    df: pd.DataFrame,
    mom_lookback: int = 126,
    vol_avg_lookback: int = 63,
    high_volume_z: float = 1.0,
    min_damp: float = 0.2,
    vol_lookback: int = 20,
    target_vol: float = 0.02,
    max_leverage: float = 1.0,
    **_,
) -> pd.Series:
    close = df["close"]
    volume = df["volume"]

    direction = np.sign(close.pct_change(mom_lookback))

    bar_return = close.pct_change()
    realized_vol = bar_return.rolling(vol_lookback).std()
    base_size = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan).clip(upper=max_leverage)

    vol_mean = volume.rolling(vol_avg_lookback).mean()
    vol_std = volume.rolling(vol_avg_lookback).std()
    volume_z = ((volume - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan)

    excess_z = (volume_z - high_volume_z).clip(lower=0.0).fillna(0.0)
    damp = (1.0 / (1.0 + excess_z)).clip(lower=min_damp, upper=1.0)

    weight = (direction * base_size * damp).clip(-1.0, 1.0)
    return weight.fillna(0.0)
