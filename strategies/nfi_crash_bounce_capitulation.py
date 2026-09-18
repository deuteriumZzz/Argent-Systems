"""Capitulation relief-rally ("crash-bounce") dip buy.

Source: reimplemented from iterativv/NostalgiaForInfinity buy_condition_170
(conceptually adapted, no code copied) — the original rule looks for a
hard crash over the prior day (large negative ROC, deeply oversold daily
RSI), but caps how deep the crash may be so it isn't buying a still-falling
knife, requires the very short-term move to still be negative (a fresh
bounce, not a rally that's already run), a volume spike on the reclaim
candle, price still near its recent low, and the first green candle that
closes back above the prior bar's high. Adapted to a single timeframe: the
source's separate daily-RSI check becomes a short-period RSI on this
dataframe's own bars.
License: GPL-3.0 (source); this reimplementation is original code
expressing the same public rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "nfi_crash_bounce_capitulation",
    "category": "mean_reversion",
    "source": "reimplemented from iterativv/NostalgiaForInfinity buy_condition_170 (conceptually adapted, no code copied)",
    "license": "GPL-3.0 (source); this reimplementation is original code expressing the same public rule",
    "description": (
        "Long entry when: ROC over `crash_lookback` bars is between "
        "`-max_crash_pct` and `-min_crash_pct` (crashed hard but not in "
        "free-fall), RSI(rsi_period) < `rsi_oversold`, the very short-term "
        "ROC over `fresh_lookback` bars is still negative (a fresh bounce, "
        "not a stale rally), volume > `vol_mult` x its rolling average, "
        "close is within `near_low_pct` of the recent low, and the candle "
        "closes green above the prior bar's high (first reclaim candle). "
        "Exits when close crosses back above the pre-crash mean."
    ),
    "default_params": {
        "crash_lookback": 96,
        "max_crash_pct": 30.0,
        "min_crash_pct": 15.0,
        "rsi_period": 3,
        "rsi_oversold": 15.0,
        "fresh_lookback": 6,
        "vol_lookback": 96,
        "vol_mult": 1.3,
        "near_low_lookback": 48,
        "near_low_pct": 0.08,
        "exit_lookback": 48,
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
    crash_lookback: int = 96,
    max_crash_pct: float = 30.0,
    min_crash_pct: float = 15.0,
    rsi_period: int = 3,
    rsi_oversold: float = 15.0,
    fresh_lookback: int = 6,
    vol_lookback: int = 96,
    vol_mult: float = 1.3,
    near_low_lookback: int = 48,
    near_low_pct: float = 0.08,
    exit_lookback: int = 48,
    **_,
) -> pd.Series:
    open_, high, close, volume = df["open"], df["high"], df["close"], df["volume"]

    crash_roc = close.pct_change(crash_lookback) * 100
    hard_crash = (crash_roc < -min_crash_pct) & (crash_roc > -max_crash_pct)

    rsi = _rsi(close, rsi_period)
    oversold = rsi < rsi_oversold

    fresh_bounce = close.pct_change(fresh_lookback) * 100 < 0

    vol_rel = volume / volume.rolling(vol_lookback).mean().replace(0, np.nan)
    vol_spike = vol_rel > vol_mult

    close_min = close.rolling(near_low_lookback).min()
    near_low = close < close_min * (1 + near_low_pct)

    reclaim_candle = (close > open_) & (close > high.shift(1))

    long_entry = hard_crash & oversold & fresh_bounce & vol_spike & near_low & reclaim_candle
    long_exit = close > close.rolling(exit_lookback).mean()

    weight = pd.Series(np.nan, index=df.index)
    # exit set first so a same-bar entry (below) can override the flatten
    weight[long_exit] = 0.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)
