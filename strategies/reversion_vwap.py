"""VWAP deviation mean reversion (RSI-filtered).

Fades price back toward a rolling volume-weighted average price once it
drops meaningfully below VWAP while momentum is oversold, and mirrors the
logic on the short side.

Source: mechanic pattern documented by the small/niche open-source repo
koopatroopa787/hermes-trading (github.com/koopatroopa787/hermes-trading,
"VWAP Mean Reversion & ORB Momentum" bot) — it enters long when price falls
below VWAP while RSI is oversold, targeting reversion back to VWAP.
Reimplemented here from that documented rule (rolling VWAP + RSI filter),
no code copied; the source repo's evolutionary parameter-mutation and
stop-loss/take-profit layers are intentionally left out to keep this a pure
vectorized signal.
License: MIT (source repo states "MIT — use at your own risk"); this file
is an independent reimplementation of the described mechanic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "reversion_vwap",
    "category": "mean_reversion",
    "source": "koopatroopa787/hermes-trading (github, VWAP Mean Reversion bot) + classic RSI filter",
    "license": "MIT (source repo); reimplemented independently",
    "description": (
        "Rolling VWAP over `vwap_period` bars. Goes long when close is "
        "below VWAP by more than `dev_pct` and RSI(rsi_period) < `oversold` "
        "(oversold dip below fair value). Exits when price closes back "
        "above VWAP. Mirrors the same logic short-side if `allow_short`."
    ),
    "default_params": {
        "vwap_period": 20,
        "dev_pct": 0.01,
        "rsi_period": 14,
        "oversold": 35,
        "overbought": 65,
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
    vwap_period: int = 20,
    dev_pct: float = 0.01,
    rsi_period: int = 14,
    oversold: float = 35,
    overbought: float = 65,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close, volume = df["close"], df["volume"]
    typical = (df["high"] + df["low"] + close) / 3
    pv = (typical * volume).rolling(vwap_period).sum()
    vol = volume.rolling(vwap_period).sum()
    vwap = pv / vol.replace(0, np.nan)

    deviation = (close - vwap) / vwap.replace(0, np.nan)
    rsi = _rsi(close, rsi_period)

    long_entries = (deviation < -dev_pct) & (rsi < oversold)
    short_entries = (deviation > dev_pct) & (rsi > overbought)
    flat = close.sub(vwap).abs() / vwap.replace(0, np.nan) < dev_pct * 0.25

    weight = pd.Series(np.nan, index=df.index)
    weight[flat] = 0.0
    weight[long_entries] = 1.0
    weight[short_entries] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
