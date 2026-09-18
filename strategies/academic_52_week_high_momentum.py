"""52-week high momentum (George & Hwang, 2004).

Source: George, T.J., Hwang, C.Y. (2004), "The 52-Week High and Momentum
Investing", Journal of Finance 59(5), 2145-2176
(https://www.bauer.uh.edu/tgeorge/papers/gh4-paper.pdf). Also a strategy
page on paperswithbacktest.com / Quantpedia
(https://quantpedia.com/strategies/52-weeks-high-effect-in-stocks/), whose
original test buys stocks/industries near their 52-week high and shorts
those far from it. Reimplemented here for a single asset (no code reused):
exposure scales continuously with how close price is to its trailing high,
the same "proximity to the 52-week high" signal used in the paper.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "academic_52_week_high_momentum",
    "category": "trend_following",
    "source": "George & Hwang (2004), 'The 52-Week High and Momentum Investing', JF 59(5)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Computes ratio = close / rolling_max(close, lookback) (the paper's "
        "PRILAG). Long-only by default: exposure ramps linearly from 0 at "
        "`threshold` to 1 at a new high (ratio=1). With allow_short, the "
        "midpoint (ratio=0.5) is used instead so being far from the high "
        "goes net short."
    ),
    "default_params": {"lookback": 252, "threshold": 0.9, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    lookback: int = 252,
    threshold: float = 0.9,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    rolling_high = close.rolling(lookback, min_periods=1).max()
    ratio = close / rolling_high

    if allow_short:
        weight = ((ratio - 0.5) * 2).clip(-1.0, 1.0)
    else:
        weight = ((ratio - threshold) / (1 - threshold)).clip(0.0, 1.0)

    return weight.fillna(0.0)
