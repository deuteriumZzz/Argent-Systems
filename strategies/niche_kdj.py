"""KDJ (stochastic + J-line) crossover.

Source: jesse-ai/example-strategies, KDJstrategy/__init__.py, credited there
to matty5690/example-strategies
(https://github.com/jesse-ai/example-strategies/blob/master/KDJstrategy/__init__.py) —
MIT licensed. KDJ itself is a stochastic-oscillator derivative popular on
Chinese/Asian retail platforms and rarely seen in Western crypto bot repos.
Reimplemented the published rule (go long while J is above both K and D,
flatten when J drops back below either) from scratch, no code copied.
License: MIT (source repo).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "niche_kdj",
    "category": "mean_reversion",
    "source": "jesse-ai/example-strategies (KDJstrategy, credited to matty5690)",
    "license": "MIT",
    "description": (
        "RSV = (close - LLV(low,period)) / (HHV(high,period) - LLV(low,period)) * 100. "
        "K/D are smoothed EMAs of RSV (alpha=1/3), J = 3K - 2D. Goes long "
        "while J is above both K and D, flat otherwise (or short if "
        "`allow_short`)."
    ),
    "default_params": {"period": 9, "allow_short": False},
}


def signals(df: pd.DataFrame, period: int = 9, allow_short: bool = False, **_) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    llv = low.rolling(period).min()
    hhv = high.rolling(period).max()
    rsv = ((close - llv) / (hhv - llv).replace(0, float("nan")) * 100).fillna(50.0)

    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    j = 3 * k - 2 * d

    bullish = (j > k) & (j > d)
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})
