"""RSI trend-consistency filter for momentum (paperswithbacktest.com).

Source: "Finding Consistent Trends with Strong Momentum - RSI for
Trend-Following and Momentum Strategies"
(https://paperswithbacktest.com/strategies/finding-consistent-trends-with-strong-momentum-rsi-for-trend-following-and-momentum-strategies).
Core idea: a raw momentum reading is a noisy, single-point signal, but an
RSI that stays parked in a moderate "trending band" (elevated but not
overbought/oversold) for many consecutive bars indicates a persistent,
low-noise trend worth sizing up - versus an RSI whipping in and out of
extremes, which flags a choppy, unreliable trend. Reimplemented here for a
single asset (no code reused): position size scales with how consistently
RSI has stayed inside its trending band over a lookback window.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_rsi_trend_consistency",
    "category": "trend_following",
    "source": "paperswithbacktest.com, 'Finding Consistent Trends with Strong Momentum - RSI for Trend-Following and Momentum Strategies'",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Computes Wilder's RSI(`rsi_period`). Bullish trending band is "
        "[`band_low`, `band_high`]; bearish mirror is [100-band_high, "
        "100-band_low]. Position size = fraction of the last `window` bars "
        "RSI spent inside the relevant band (a 0-1 consistency score), "
        "directed long when RSI > 50 (short when < 50 if allow_short)."
    ),
    "default_params": {
        "rsi_period": 14,
        "band_low": 45.0,
        "band_high": 75.0,
        "window": 20,
        "allow_short": False,
    },
}


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def signals(
    df: pd.DataFrame,
    rsi_period: int = 14,
    band_low: float = 45.0,
    band_high: float = 75.0,
    window: int = 20,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    rsi = _rsi(close, rsi_period)

    bullish_band = rsi.between(band_low, band_high)
    bullish_consistency = bullish_band.rolling(window).mean()

    weight = pd.Series(0.0, index=df.index)
    bullish = rsi > 50
    weight = weight.where(~bullish, bullish_consistency)

    if allow_short:
        bearish_band = rsi.between(100 - band_high, 100 - band_low)
        bearish_consistency = bearish_band.rolling(window).mean()
        weight = weight.where(bullish, -bearish_consistency)

    return weight.fillna(0.0).clip(-1.0, 1.0)
