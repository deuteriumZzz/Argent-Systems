"""Bullish engulfing candle as a trend-continuation trigger.

Source: reimplemented from iterativv/NostalgiaForInfinity buy_condition_167
(conceptually adapted, no code copied) — the original rule requires the
higher (daily) timeframe not be floored, short-term momentum already
pointed up but the higher (4h) timeframe not yet overheated, an established
uptrend regime (close above EMA200 and RSI above 50), and the trigger
candle itself forms a bullish engulfing pattern. Adapted to a single
timeframe by using RSI/ROC computed on this dataframe's own bars in place
of the source's separate 1h/4h/1d checks.
License: GPL-3.0 (source); this reimplementation is original code
expressing the same public rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "nfi_engulfing_trend_continuation",
    "category": "trend_following",
    "source": "reimplemented from iterativv/NostalgiaForInfinity buy_condition_167 (conceptually adapted, no code copied)",
    "license": "GPL-3.0 (source); this reimplementation is original code expressing the same public rule",
    "description": (
        "Long entry on a bullish engulfing candle (this candle's body "
        "fully covers and reverses the prior candle's body) when close > "
        "EMA(trend_ema_period), RSI(rsi_period) > `rsi_min`, short-term "
        "momentum (RSI(fast_rsi_period)) > `fast_rsi_min`, and ROC over "
        "`roc_period` bars < `roc_ceiling` (not already overheated). Exits "
        "when close falls back below the trend EMA."
    ),
    "default_params": {
        "trend_ema_period": 200,
        "rsi_period": 14,
        "rsi_min": 50.0,
        "fast_rsi_period": 3,
        "fast_rsi_min": 55.0,
        "roc_period": 24,
        "roc_ceiling": 8.0,
        "allow_short": False,
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
    trend_ema_period: int = 200,
    rsi_period: int = 14,
    rsi_min: float = 50.0,
    fast_rsi_period: int = 3,
    fast_rsi_min: float = 55.0,
    roc_period: int = 24,
    roc_ceiling: float = 8.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    open_, close = df["open"], df["close"]
    prev_open, prev_close = open_.shift(1), close.shift(1)

    bullish_engulf = (
        (close > open_)
        & (prev_close < prev_open)
        & (close >= prev_open)
        & (open_ <= prev_close)
    )
    bearish_engulf = (
        (close < open_)
        & (prev_close > prev_open)
        & (close <= prev_open)
        & (open_ >= prev_close)
    )

    ema = close.ewm(span=trend_ema_period, adjust=False).mean()
    uptrend = close > ema
    downtrend = close < ema

    rsi = _rsi(close, rsi_period)
    fast_rsi = _rsi(close, fast_rsi_period)
    roc = close.pct_change(roc_period) * 100

    long_entry = (
        bullish_engulf & uptrend & (rsi > rsi_min) & (fast_rsi > fast_rsi_min) & (roc < roc_ceiling)
    )

    weight = pd.Series(np.nan, index=df.index)
    # trend flips first so a same-bar entry (below) can override the flatten
    weight[~uptrend] = 0.0
    if allow_short:
        weight[~downtrend] = 0.0
        short_entry = (
            bearish_engulf
            & downtrend
            & (rsi < (100 - rsi_min))
            & (fast_rsi < (100 - fast_rsi_min))
            & (roc > -roc_ceiling)
        )
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)
