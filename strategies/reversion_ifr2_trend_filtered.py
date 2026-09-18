"""RSI(2) mean reversion filtered by a higher-timeframe-style trend gate.

Only takes the RSI(2) oversold dip when the broader trend (EMA slope) is
still up, and exits on the same "close above the recent 2-bar high" rule
as the source strategy — avoiding buying dips inside an established
downtrend.

Source: mechanic pattern from the small open-source repo
jesse-ai/example-strategies, strategy "IFR2"
(github.com/jesse-ai/example-strategies/blob/master/IFR2/__init__.py) —
goes long on RSI(2) < 10 filtered by an uptrend/cloud-based trend check,
and exits when close exceeds the highest close of the prior two candles.
Reimplemented here with an EMA-slope trend filter standing in for that
strategy's Ichimoku-cloud + Hilbert-Transform trend-mode filters (dropped
to keep this a single self-contained vectorized signal with no extra
indicator stack), no code copied.
License: not specified by the source repo; this file is an independent
reimplementation of the described mechanic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "reversion_ifr2_trend_filtered",
    "category": "mean_reversion",
    "source": "jesse-ai/example-strategies IFR2 mechanic (github), reimplemented with an EMA trend filter",
    "license": "unspecified upstream; independent reimplementation",
    "description": (
        "Goes long when RSI(rsi_period) < `oversold` AND the trend EMA is "
        "rising (trend_ema > trend_ema shifted `slope_lookback` bars back). "
        "Exits when close exceeds the highest close of the prior "
        "`exit_lookback` bars. No short side (long-only dip buy, matching "
        "the source strategy)."
    ),
    "default_params": {
        "rsi_period": 2,
        "oversold": 10,
        "trend_ema_period": 50,
        "slope_lookback": 5,
        "exit_lookback": 2,
    },
}


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def signals(
    df: pd.DataFrame,
    rsi_period: int = 2,
    oversold: float = 10,
    trend_ema_period: int = 50,
    slope_lookback: int = 5,
    exit_lookback: int = 2,
    **_,
) -> pd.Series:
    close = df["close"]
    rsi = _rsi(close, rsi_period)
    trend_ema = close.ewm(span=trend_ema_period, adjust=False).mean()
    uptrend = trend_ema > trend_ema.shift(slope_lookback)

    entries = (rsi < oversold) & uptrend
    exits = close > close.shift(1).rolling(exit_lookback).max()

    weight = pd.Series(np.nan, index=df.index)
    weight[entries] = 1.0
    weight[exits] = 0.0
    return weight.ffill().fillna(0.0)
