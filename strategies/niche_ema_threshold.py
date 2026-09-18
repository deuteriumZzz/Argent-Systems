"""Very-long EMA crossover with an asymmetric exit hysteresis band.

Source: paulcpk/freqtrade-strategies-that-work, EMAPriceCrossoverWithThreshold.py
(https://github.com/paulcpk/freqtrade-strategies-that-work/blob/main/EMAPriceCrossoverWithThreshold.py) —
MIT licensed. The mechanic: use one very slow EMA (period 800 on 1h in the
original) as the trend line, enter when price crosses above it, but only
exit once price crosses below a *lower* threshold line
(ema * (1 - threshold_pct)) instead of the EMA itself — a deliberate dead
zone that avoids exiting on noise right at the EMA. Reimplemented the
published rule from scratch (original is a freqtrade IStrategy class), no
code copied.
License: MIT (source repo).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "niche_ema_threshold",
    "category": "trend_following",
    "source": "paulcpk/freqtrade-strategies-that-work (EMAPriceCrossoverWithThreshold)",
    "license": "MIT",
    "description": (
        "Long entry when close crosses above EMA(ema_period). Exit only "
        "when close crosses below EMA(ema_period) * (1 - threshold_pct), a "
        "band below the EMA rather than the EMA itself, to avoid whipsaw "
        "exits right at the line."
    ),
    "default_params": {"ema_period": 200, "threshold_pct": 0.01},
}


def signals(
    df: pd.DataFrame, ema_period: int = 200, threshold_pct: float = 0.01, **_
) -> pd.Series:
    close = df["close"]
    ema = close.ewm(span=ema_period, adjust=False).mean()
    exit_line = ema * (1 - threshold_pct)

    entries = (close > ema) & (close.shift(1) <= ema.shift(1))
    exits = (close < exit_line) & (close.shift(1) >= exit_line.shift(1))

    weight = pd.Series(float("nan"), index=df.index)
    weight[entries] = 1.0
    weight[exits] = 0.0
    return weight.ffill().fillna(0.0)
