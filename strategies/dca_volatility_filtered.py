"""Volatility-filtered DCA — skips a scheduled buy when short-term
volatility (ATR as a fraction of price) is too high.

Source: pattern documented by 3Commas DCA Bot's indicator-gated "Start
Conditions" (a deal/safety order only fires if the configured technical
condition is met) — applied here with a volatility gate instead of a
directional indicator. See:
https://help.3commas.io/en/articles/3108986-dca-bot-start-close-conditions-via-indicators
Reimplemented from the publicly documented mechanic, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "dca_volatility_filtered",
    "category": "accumulation",
    "source": "3Commas DCA Bot (indicator-gated start/safety-order conditions)",
    "license": "N/A",
    "description": (
        "Periodic DCA buy every `interval_bars`, deferred bar-by-bar "
        "whenever ATR(atr_period) as a fraction of price exceeds "
        "`max_volatility_pct` — avoids adding exposure during violent, "
        "high-ATR stretches and waits for calmer conditions."
    ),
    "default_params": {
        "interval_bars": 24 * 3,
        "num_buys": 20,
        "atr_period": 14,
        "max_volatility_pct": 0.05,
    },
}


def _atr_pct(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    return atr / close


def signals(
    df: pd.DataFrame,
    interval_bars: int = 24 * 3,
    num_buys: int = 20,
    atr_period: int = 14,
    max_volatility_pct: float = 0.05,
    **_,
) -> pd.Series:
    atr_pct = _atr_pct(df, atr_period).to_numpy()
    n = len(df)
    step = 1.0 / num_buys

    weight = pd.Series(0.0, index=df.index)
    buys_done = 0
    next_check = 0
    while next_check < n and buys_done < num_buys:
        v = atr_pct[next_check]
        if np.isnan(v) or v <= max_volatility_pct:
            buys_done += 1
            weight.iloc[next_check:] = min(1.0, buys_done * step)
            next_check += interval_bars
        else:
            next_check += 1
    return weight
