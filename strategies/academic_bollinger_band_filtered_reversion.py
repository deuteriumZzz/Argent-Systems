"""Bollinger Bands with a re-entry confirmation filter (Lento et al., 2007).

Source: Lento, C., Gradojevic, N. & Wright, C.S. (2007), "Investment
Information Content in Bollinger Bands?", Applied Financial Economics
Letters 3(4), 263-267 - tests Bollinger Bands as a technical trading
signal using a band-touch-then-turn confirmation filter rather than a
raw threshold-crossing rule. Distinct from reversion_bollinger_percent_b.py
(which enters as soon as %B crosses a static threshold): this only enters
once price has closed back inside the band after having closed outside it -
the specific "re-entry" confirmation the paper tests - and exits at the
middle band (the moving average) rather than a %B midline level. No code
reused, only the documented rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_bollinger_band_filtered_reversion",
    "category": "mean_reversion",
    "source": "Lento, Gradojevic & Wright (2007), 'Investment Information Content in Bollinger Bands?', Applied Financial Economics Letters 3(4)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Goes long on the bar where close re-enters the band from below "
        "(previous close was under the lower band, current close is back "
        "at/above it); mirrors for short re-entry from above the upper band. "
        "Exits (flattens) on the bar the close crosses the middle band "
        "(the rolling moving average)."
    ),
    "default_params": {"period": 20, "num_std": 2.0, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std

    reentry_long = (close.shift(1) < lower.shift(1)) & (close >= lower)
    reentry_short = (close.shift(1) > upper.shift(1)) & (close <= upper)
    crossed_up = (close.shift(1) < mid.shift(1)) & (close >= mid)
    crossed_down = (close.shift(1) > mid.shift(1)) & (close <= mid)
    touch_mid = crossed_up | crossed_down

    weight = pd.Series(np.nan, index=df.index)
    weight[touch_mid] = 0.0
    weight[reentry_long] = 1.0
    weight[reentry_short] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
