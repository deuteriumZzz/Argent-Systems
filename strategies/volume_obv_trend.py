"""On-Balance Volume (OBV) trend-following crossover.

Source: classic concept (Joseph Granville's On-Balance Volume, 1963;
documented formula as used by StockCharts/Investopedia). OBV is a running
cumulative total of volume, added on up-close bars and subtracted on
down-close bars — the idea being that volume leads price. This strategy
treats OBV itself as a trend signal by comparing it to its own moving
average (a rising OBV above its average confirms accumulation supporting
the trend), rather than combining it with price oscillators as elsewhere
in this repo. Reimplemented from the publicly documented formula, no code
borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "volume_obv_trend",
    "category": "trend_following",
    "source": "classic concept (Joseph Granville's On-Balance Volume)",
    "license": "N/A",
    "description": (
        "OBV = cumulative sum of volume, signed by the direction of each "
        "bar's close-to-close change. Long while OBV is above its own "
        "`ma_period`-bar SMA (volume flow confirms the up-move), flat (or "
        "short if `allow_short`) while below."
    ),
    "default_params": {"ma_period": 20, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    ma_period: int = 20,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close, volume = df["close"], df["volume"]

    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    obv_ma = obv.rolling(ma_period).mean()

    short_weight = -1.0 if allow_short else 0.0
    return (obv > obv_ma).map({True: 1.0, False: short_weight})
