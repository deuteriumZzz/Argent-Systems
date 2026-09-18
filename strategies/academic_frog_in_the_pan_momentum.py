"""Frog-in-the-pan continuous-information momentum (Da, Gurun & Warachka, 2014).

Source: Da, Z., Gurun, U.G., Warachka, M. (2014), "Frog in the Pan:
Continuous Information and Momentum", Review of Financial Studies 27(7),
2171-2218 (https://doi.org/10.1093/rfs/hhu003). Key finding: momentum
driven by many small, same-direction daily moves ("continuous"
information diffusion, low "information discreteness") persists and keeps
paying off, while momentum built from a few large jumps ("discrete"
information) tends to reverse quickly - investors underreact to gradual
news but overreact to salient jumps. Their information discreteness (ID)
measure is sign(return) x (%negative days - %positive days) over the
formation window. Reimplemented here for a single asset (no code reused):
standard trailing-return momentum, sized up when the trend has been
continuous (low ID) and sized down when it has been jumpy/discrete
(high ID).
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_frog_in_the_pan_momentum",
    "category": "trend_following",
    "source": "Da, Gurun & Warachka (2014), 'Frog in the Pan: Continuous Information and Momentum', RFS 27(7)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Base signal: sign of `mom_lookback`-bar trailing return, inverse-"
        "vol sized. Computes the paper's information-discreteness measure "
        "ID = sign(return) x (%negative_days - %positive_days) over the "
        "same window; low/negative ID (gradual, same-direction drift) "
        "raises conviction toward 1, high ID (a few large jumps) lowers it "
        "toward `min_conviction`, via conviction = clip(0.5 - "
        "id_scale*ID, min_conviction, 1)."
    ),
    "default_params": {
        "mom_lookback": 126,
        "id_scale": 5.0,
        "min_conviction": 0.2,
        "vol_lookback": 20,
        "target_vol": 0.02,
        "max_leverage": 1.0,
    },
}


def signals(
    df: pd.DataFrame,
    mom_lookback: int = 126,
    id_scale: float = 5.0,
    min_conviction: float = 0.2,
    vol_lookback: int = 20,
    target_vol: float = 0.02,
    max_leverage: float = 1.0,
    **_,
) -> pd.Series:
    close = df["close"]
    trailing_return = close.pct_change(mom_lookback)
    direction = np.sign(trailing_return)

    bar_return = close.pct_change()
    pct_pos = (bar_return > 0).rolling(mom_lookback).mean()
    pct_neg = (bar_return < 0).rolling(mom_lookback).mean()
    information_discreteness = direction * (pct_neg - pct_pos)

    conviction = (0.5 - id_scale * information_discreteness).clip(lower=min_conviction, upper=1.0)

    realized_vol = bar_return.rolling(vol_lookback).std()
    base_size = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan).clip(upper=max_leverage)

    weight = (direction * base_size * conviction).clip(-1.0, 1.0)
    return weight.fillna(0.0)
