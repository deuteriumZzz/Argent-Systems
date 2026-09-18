"""Parabolic SAR trend flip.

Source: classic concept (J. Welles Wilder's Parabolic SAR, "New Concepts in
Technical Trading Systems", 1978). A staple trailing-stop-and-reverse
mechanic reused across countless public crypto bot repos. Reimplemented from
the published recursive formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "trend_parabolic_sar",
    "category": "trend_following",
    "source": "classic concept (Wilder's Parabolic SAR)",
    "license": "N/A",
    "description": (
        "Long while price is above the Parabolic SAR dot (uptrend), flat "
        "(or short if `allow_short`) while below. Acceleration factor "
        "ratchets up each new extreme, as in the original formula."
    ),
    "default_params": {
        "af_start": 0.02,
        "af_step": 0.02,
        "af_max": 0.2,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)

    trend = np.ones(n)  # 1 = long, -1 = short
    ep = high[0]
    af = af_start
    sar = np.zeros(n)
    sar[0] = low[0]

    for i in range(1, n):
        prev_sar = sar[i - 1]
        prior_extreme_low = low[i - 2] if i >= 2 else low[i - 1]
        prior_extreme_high = high[i - 2] if i >= 2 else high[i - 1]

        if trend[i - 1] == 1:
            candidate = prev_sar + af * (ep - prev_sar)
            candidate = min(candidate, low[i - 1], prior_extreme_low)
            if low[i] < candidate:
                trend[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                trend[i] = 1
                sar[i] = candidate
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            candidate = prev_sar + af * (ep - prev_sar)
            candidate = max(candidate, high[i - 1], prior_extreme_high)
            if high[i] > candidate:
                trend[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                trend[i] = -1
                sar[i] = candidate
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    short_weight = -1.0 if allow_short else 0.0
    return pd.Series(trend, index=df.index).map({1: 1.0, -1: short_weight})
