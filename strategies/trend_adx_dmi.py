"""ADX/DMI trend-strength filter combined with moving-average direction.

Source: classic concept (J. Welles Wilder's ADX / +DI / -DI, "New Concepts in
Technical Trading Systems", 1978) used as a trend-strength gate on top of a
simple fast/slow moving-average direction filter — a very common combination
in public crypto trend-following bots. Reimplemented from the published
Wilder smoothing formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "trend_adx_dmi",
    "category": "trend_following",
    "source": "classic concept (Wilder's ADX/DMI as a trend-strength filter)",
    "license": "N/A",
    "description": (
        "Only takes a directional position while ADX(period) is above "
        "`adx_threshold` (a strong trend is in force); direction is set by "
        "fast SMA vs slow SMA. Flat whenever ADX is below the threshold."
    ),
    "default_params": {
        "period": 14,
        "adx_threshold": 20.0,
        "fast": 10,
        "slow": 30,
        "allow_short": False,
    },
}


def _dmi_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def signals(
    df: pd.DataFrame,
    period: int = 14,
    adx_threshold: float = 20.0,
    fast: int = 10,
    slow: int = 30,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    adx = _dmi_adx(df, period)
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()

    trending = adx > adx_threshold
    bullish = fast_ma > slow_ma

    short_weight = -1.0 if allow_short else 0.0
    weight = pd.Series(0.0, index=df.index)
    weight[trending & bullish] = 1.0
    weight[trending & ~bullish] = short_weight
    return weight
