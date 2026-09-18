"""Dual Thrust range breakout (Michael Chalek).

Source: jesse-ai/example-strategies, DUAL_THRUST/__init__.py
(https://github.com/jesse-ai/example-strategies/blob/master/DUAL_THRUST/__init__.py) —
MIT licensed. Reimplemented from that file's documented rule (range
computed from N-bar highs/lows/closes, asymmetric up/down coefficients,
breakout of the day's open +/- range), no code copied — the original is
Jesse-framework event-driven Strategy class methods; this is a from-scratch
vectorized pandas rewrite of the same published mechanic (itself credited
there to a Medium write-up of Michael Chalek's original "Dual Thrust" system).
License: MIT (source repo).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "niche_dual_thrust",
    "category": "trend_following",
    "source": "jesse-ai/example-strategies (DUAL_THRUST) - Michael Chalek's Dual Thrust",
    "license": "MIT",
    "description": (
        "Range = max(HH(period)-LC(period), HC(period)-LL(period)) computed "
        "from the prior `period` bars. Goes long when close breaks above "
        "open + k1*range, short (or flat) when close breaks below "
        "open - k2*range. Position held until the opposite breakout fires."
    ),
    "default_params": {"period": 20, "k1": 0.5, "k2": 0.5, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    period: int = 20,
    k1: float = 0.5,
    k2: float = 0.5,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]

    hh = high.rolling(period).max().shift(1)
    hc = close.rolling(period).max().shift(1)
    ll = low.rolling(period).min().shift(1)
    lc = close.rolling(period).min().shift(1)

    rng = pd.concat([hh - lc, hc - ll], axis=1).max(axis=1)
    upper = open_ + k1 * rng
    lower = open_ - k2 * rng

    long_break = close > upper
    short_break = close < lower

    weight = pd.Series(float("nan"), index=df.index)
    weight[long_break] = 1.0
    weight[short_break] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
