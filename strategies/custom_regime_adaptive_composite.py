"""Regime-Adaptive Composite: switches between breakout, capitulation-cash,
recovery re-entry, and range accumulation based on the detected regime.

Original idea, synthesized directly from this project's own cross-regime
robustness results (results/regime_comparison_robustness.csv over 11 BTC
market regimes, 2017-2025):
  - niche_dual_thrust (breakout) wins in clean trends (bull_2023_2024_etf
    3.38, bull_2021_q4 2.98) but loses in slow grinds (correction_2021
    -2.13, bear_2018 -1.09).
  - grid/dca strategies win in range/accumulation regimes but lose sharply
    in fast crashes (covid_crash_2020, correction_2021).
  - onchain_hash_ribbon pinpoints the exact capitulation -> recovery turn
    (the entry into bull_2020_2021 and the ETF rally).
None of those three failure modes overlap, so this strategy tries to route
around each one instead of eating any single one's worst regime:
  1. Active capitulation + spiking volatility -> flat (don't catch the
     falling knife the way plain DCA does).
  2. Hash Ribbon just flipped out of capitulation -> aggressive long (the
     highest historical risk/reward window).
  3. A Dual-Thrust-style range breakout -> follow the trend.
  4. Low-volatility, non-trending -> modest range exposure.
No external repo copied; the breakout/range logic below is a light,
self-contained reimplementation of ideas this project already validated
elsewhere (niche_dual_thrust.py, onchain_hash_ribbon.py), kept standalone
per this repo's own contract of not importing between strategy files.
License: N/A (custom idea, unvalidated — this is a first cut, not a
validated edge; see the backtest results before trusting it with capital).

DIFFERENT DATA CONTRACT: requires the `hash_rate` on-chain column merged
onto BTC OHLCV — see backtest/run_onchain.py. META["data_type"] = "onchain".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "custom_regime_adaptive_composite",
    "category": "composite",
    "data_type": "onchain",
    "source": (
        "Original composite, synthesized from this project's own cross-regime "
        "robustness findings (niche_dual_thrust trend edge, grid/dca range edge, "
        "onchain_hash_ribbon capitulation-turn detection)"
    ),
    "license": "N/A (custom idea, unvalidated)",
    "description": (
        "Flat during active hash-rate capitulation with spiking volatility; "
        "aggressively long right after Hash Ribbon exits capitulation; "
        "follows Dual-Thrust-style range breakouts in trending conditions; "
        "modest exposure in calm, non-trending ranges."
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
        "recovery_window": 14,
    },
}


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
    recovery_window: int = 14,
    **_,
) -> pd.Series:
    # --- Hash Ribbon regime (see onchain_hash_ribbon.py for the standalone version) ---
    fast_hash = df["hash_rate"].rolling(hash_fast).mean()
    slow_hash = df["hash_rate"].rolling(hash_slow).mean()
    capitulation = fast_hash < slow_hash
    just_recovered = (fast_hash > slow_hash) & (fast_hash.shift(1) <= slow_hash.shift(1))
    recovery_recent = (
        just_recovered.astype(int).rolling(recovery_window, min_periods=1).max().astype(bool)
    )

    # --- Realized-volatility regime ---
    returns = df["close"].pct_change()
    realized_vol = returns.rolling(vol_window).std()
    vol_mean = realized_vol.rolling(vol_lookback).mean()
    vol_std = realized_vol.rolling(vol_lookback).std()
    vol_zscore = (realized_vol - vol_mean) / vol_std.replace(0, np.nan)
    crash_vol = vol_zscore >= vol_spike_threshold

    # --- Dual-Thrust-style range breakout (see niche_dual_thrust.py) ---
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

    low_vol_range = (vol_zscore < 0) & ~breakout_long & ~breakout_short & ~crash_vol

    conditions = [
        crash_vol & capitulation,
        recovery_recent,
        breakout_long,
        breakout_short,
        low_vol_range,
    ]
    choices = [0.0, 1.0, 1.0, -1.0, 0.5]

    weight = pd.Series(np.select(conditions, choices, default=np.nan), index=df.index)
    return weight.ffill().fillna(0.0)
