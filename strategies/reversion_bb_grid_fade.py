"""Bollinger Bands mean-reversion grid (scaled fade with a middle-band target).

Instead of a single all-in/all-out fade, this scales exposure in as price
pushes further outside the bands (further overextension = bigger fade bet,
capped at full size) and scales back out as price reverts toward the
midline, flattening once it's back inside a tight band around the mean.

Source: mechanic pattern documented by the small/niche open-source repo
luximil/Bollinger-Bands-Mean-Reversion-Grid
(github.com/luximil/Bollinger-Bands-Mean-Reversion-Grid) — trades price's
reversion to the moving average once it surpasses an outer Bollinger band,
using a set of graduated positions closed at intermediate levels between
the outer band and the center line, originally aimed at correlated FX
pairs. Reimplemented here as a single-asset graduated fade (no pairs leg,
no exact position-count/spacing from the source), no code copied.
License: not specified by the source repo; this file is an independent
reimplementation of the described mechanic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "reversion_bb_grid_fade",
    "category": "mean_reversion",
    "source": "luximil/Bollinger-Bands-Mean-Reversion-Grid (github) mechanic, reimplemented",
    "license": "unspecified upstream; independent reimplementation",
    "description": (
        "Bollinger Bands over `period`/`num_std`. Position size scales "
        "linearly with how far %B sits outside [0, 1], capped at full size, "
        "fading toward the midline. Flattens once %B is back within "
        "`flat_band` of 0.5."
    ),
    "default_params": {
        "period": 20,
        "num_std": 2.0,
        "max_overshoot": 0.5,
        "flat_band": 0.05,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    max_overshoot: float = 0.5,
    flat_band: float = 0.05,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    percent_b = (close - lower) / (upper - lower).replace(0, np.nan)

    # overshoot below 0 (oversold) -> positive long size; above 1 (overbought) -> short size
    overshoot_low = (-percent_b).clip(lower=0)
    overshoot_high = (percent_b - 1).clip(lower=0)
    long_size = (overshoot_low / max_overshoot).clip(upper=1.0)
    short_size = (overshoot_high / max_overshoot).clip(upper=1.0)

    weight = long_size - (short_size if allow_short else 0.0)
    weight = weight.where(percent_b.sub(0.5).abs() > flat_band, 0.0)
    return weight.clip(-1.0, 1.0).fillna(0.0)
