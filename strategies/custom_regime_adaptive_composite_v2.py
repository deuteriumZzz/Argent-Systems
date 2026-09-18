"""Regime-Adaptive Composite v2: simplified after v1's walk-forward tuning
failed to generalize (see results/composite_tuning_top10_oos.csv — the
best in-sample parameter combos scored ~1.0 train Sharpe but only 0.07-0.29
out-of-sample, barely above the untuned defaults). That result says the
architecture, not the thresholds, was the problem — so this version
removes the two most discretionary, least-validated pieces of v1 instead
of re-tuning them:

  - Dropped the flat 0.5 "low-volatility range" bet: it had no directional
    logic at all, just a blind half-position, and it's the main thing
    dragging down sideways_2023 / bear_2022 in v1's regime results.
  - Dropped the "just exited capitulation -> aggressive long for N days"
    special case: it only fires a handful of times across all of BTC's
    history, which isn't enough occurrences to trust a fitted rule around.
  - Added an ADX trend-strength filter (same concept as trend_adx_dmi.py,
    reimplemented standalone here per this repo's no-cross-imports rule):
    the Dual-Thrust-style breakout is only taken when ADX confirms an
    actual trend is present, instead of firing on any range breakout
    including ones inside a choppy, non-trending market.

Net effect: three states instead of five (capitulation-flat, trending
breakout long/short, otherwise flat) — fewer discretionary branches, each
with a clearer causal story, on the theory that a simpler rule generalizes
better than a more elaborate one tuned to fit the past (exactly what v1's
own tuning run demonstrated).
License: N/A (custom idea, unvalidated — compare directly against v1 and
the untuned defaults before trusting either).

DIFFERENT DATA CONTRACT: requires the `hash_rate` on-chain column merged
onto BTC OHLCV — see backtest/run_onchain.py. META["data_type"] = "onchain".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "custom_regime_adaptive_composite_v2",
    "category": "composite",
    "data_type": "onchain",
    "source": (
        "Original composite, v2 simplification of custom_regime_adaptive_composite.py "
        "after its walk-forward tuning run (results/composite_tuning.csv) showed the "
        "5-branch v1 architecture didn't generalize out-of-sample"
    ),
    "license": "N/A (custom idea, unvalidated)",
    "description": (
        "Flat during active hash-rate capitulation with spiking volatility; "
        "otherwise follows Dual-Thrust-style range breakouts, but only when "
        "ADX confirms a real trend is present; flat everywhere else "
        "(no blind range-exposure bet, no special-cased recovery boost)."
    ),
    "default_params": {
        "hash_fast": 30,
        "hash_slow": 60,
        "vol_window": 14,
        "vol_lookback": 90,
        "vol_spike_threshold": 1.5,
        "range_period": 3,
        "k1": 0.5,
        "k2": 0.5,
        "adx_period": 14,
        "adx_threshold": 20,
    },
}


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )
    true_range = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)

    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def signals(
    df: pd.DataFrame,
    hash_fast: int = 30,
    hash_slow: int = 60,
    vol_window: int = 14,
    vol_lookback: int = 90,
    vol_spike_threshold: float = 1.5,
    range_period: int = 3,
    k1: float = 0.5,
    k2: float = 0.5,
    adx_period: int = 14,
    adx_threshold: float = 20,
    **_,
) -> pd.Series:
    # --- Capitulation circuit-breaker (kept from v1 — a genuine, rare, extreme-event guard) ---
    fast_hash = df["hash_rate"].rolling(hash_fast).mean()
    slow_hash = df["hash_rate"].rolling(hash_slow).mean()
    capitulation = fast_hash < slow_hash

    returns = df["close"].pct_change()
    realized_vol = returns.rolling(vol_window).std()
    vol_mean = realized_vol.rolling(vol_lookback).mean()
    vol_std = realized_vol.rolling(vol_lookback).std()
    vol_zscore = (realized_vol - vol_mean) / vol_std.replace(0, np.nan)
    crash_vol = vol_zscore >= vol_spike_threshold

    # --- ADX trend-strength gate (new in v2) ---
    trending = _adx(df, adx_period) >= adx_threshold

    # --- Dual-Thrust-style range breakout direction ---
    recent_high = df["high"].rolling(range_period).max()
    recent_low = df["low"].rolling(range_period).min()
    recent_close_high = df["close"].rolling(range_period).max()
    recent_close_low = df["close"].rolling(range_period).min()
    range_width = pd.concat(
        [recent_high - recent_close_low, recent_close_high - recent_low], axis=1
    ).max(axis=1)
    upper_trigger = df["open"] + k1 * range_width.shift(1)
    lower_trigger = df["open"] - k2 * range_width.shift(1)
    breakout_long = df["close"] > upper_trigger
    breakout_short = df["close"] < lower_trigger

    conditions = [
        crash_vol & capitulation,
        trending & breakout_long,
        trending & breakout_short,
    ]
    choices = [0.0, 1.0, -1.0]

    weight = pd.Series(np.select(conditions, choices, default=np.nan), index=df.index)
    return weight.ffill().fillna(0.0)
