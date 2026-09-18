"""MFI + CMF accumulation confirmation at the Bollinger lower band.

Source: reimplemented from iterativv/NostalgiaForInfinity buy_condition_11
(conceptually adapted, no code copied) — the original rule (labelled
"MFI/CMF Double Divergence + BB Reversion") requires both a volume-flow
oscillator (MFI) and a volume-weighted accumulation/distribution oscillator
(CMF) to read positive (buyers still active despite the price dip) while
price sits at/below the Bollinger lower band and Williams %R confirms a
deep oversold read on two different lookbacks — i.e. the drop is being
absorbed, not sold into. Adapted to a single timeframe by dropping the
source's separate 1h Williams %R check in favor of one extra Williams %R
lookback on this dataframe.
License: GPL-3.0 (source); this reimplementation is original code
expressing the same public rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "nfi_mfi_cmf_bb_accumulation",
    "category": "accumulation",
    "source": "reimplemented from iterativv/NostalgiaForInfinity buy_condition_11 (conceptually adapted, no code copied)",
    "license": "GPL-3.0 (source); this reimplementation is original code expressing the same public rule",
    "description": (
        "Long entry when close <= `bb_band_mult` x the Bollinger lower "
        "band (period, num_std), MFI(mfi_period) > `mfi_min` and "
        "CMF(cmf_period) > `cmf_min` (buyers still accumulating despite "
        "the dip), and Williams %R over both `willr_fast_period` and "
        "`willr_slow_period` reads below `willr_oversold` (confirmed "
        "oversold on two lookbacks). Exits when close crosses back above "
        "the Bollinger midline."
    ),
    "default_params": {
        "bb_period": 20,
        "bb_num_std": 2.0,
        "bb_band_mult": 1.015,
        "mfi_period": 14,
        "mfi_min": 30.0,
        "cmf_period": 20,
        "cmf_min": 0.02,
        "willr_fast_period": 14,
        "willr_slow_period": 48,
        "willr_oversold": -80.0,
        "allow_short": False,
    },
}


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    typical = (high + low + close) / 3
    raw_flow = typical * volume
    rising = typical > typical.shift(1)
    pos_flow = raw_flow.where(rising, 0.0).rolling(period).sum()
    neg_flow = raw_flow.where(~rising, 0.0).rolling(period).sum()
    money_ratio = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - (100 / (1 + money_ratio))


def _cmf(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = (mfm * volume).fillna(0.0)
    return mfv.rolling(period).sum() / volume.rolling(period).sum().replace(0, np.nan)


def _willr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    span = (highest_high - lowest_low).replace(0, np.nan)
    return -100 * (highest_high - close) / span


def signals(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_num_std: float = 2.0,
    bb_band_mult: float = 1.015,
    mfi_period: int = 14,
    mfi_min: float = 30.0,
    cmf_period: int = 20,
    cmf_min: float = 0.02,
    willr_fast_period: int = 14,
    willr_slow_period: int = 48,
    willr_oversold: float = -80.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]

    bb_mid = close.rolling(bb_period).mean()
    bb_stdev = close.rolling(bb_period).std()
    bb_lower = bb_mid - bb_num_std * bb_stdev
    bb_upper = bb_mid + bb_num_std * bb_stdev
    at_lower_band = close <= bb_lower * bb_band_mult

    mfi = _mfi(high, low, close, volume, mfi_period)
    cmf = _cmf(high, low, close, volume, cmf_period)
    accumulating = (mfi > mfi_min) & (cmf > cmf_min)

    willr_fast = _willr(high, low, close, willr_fast_period)
    willr_slow = _willr(high, low, close, willr_slow_period)
    double_oversold = (willr_fast < willr_oversold) & (willr_slow < willr_oversold)

    long_entry = at_lower_band & accumulating & double_oversold
    long_exit = close > bb_mid

    weight = pd.Series(np.nan, index=df.index)
    # exit set first so a same-bar entry (below) can override the flatten
    weight[long_exit] = 0.0
    if allow_short:
        at_upper_band = close >= bb_upper / bb_band_mult
        distributing = (mfi < (100 - mfi_min)) & (cmf < -cmf_min)
        double_overbought = (willr_fast > -willr_oversold) & (willr_slow > -willr_oversold)
        short_entry = at_upper_band & distributing & double_overbought
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)
