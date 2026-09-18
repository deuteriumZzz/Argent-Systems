"""Williams %R oversold/overbought mean reversion.

Source: classic technical-analysis concept, developed by Larry Williams
(described in his published trading writings; ubiquitous in TA references
and crypto bot repos as a bounded momentum oscillator). Reimplemented from
the published formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "reversion_williams_r",
    "category": "mean_reversion",
    "source": "classic concept (Larry Williams' %R)",
    "license": "N/A",
    "description": (
        "Williams %R over `period` bars (range -100..0). Goes long when "
        "%R < `oversold` (near -100, deeply oversold), exits/shorts when "
        "%R > `overbought` (near 0, deeply overbought)."
    ),
    "default_params": {
        "period": 14,
        "oversold": -80,
        "overbought": -20,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = -80,
    overbought: float = -20,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    span = (highest_high - lowest_low).replace(0, np.nan)
    percent_r = -100 * (highest_high - close) / span

    long_entries = percent_r < oversold
    short_entries = percent_r > overbought

    weight = pd.Series(np.nan, index=df.index)
    weight[long_entries] = 1.0
    weight[short_entries] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
