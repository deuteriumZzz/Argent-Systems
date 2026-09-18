"""MACD crossover trend-following.

Source: classic technical-analysis concept (Gerald Appel, MACD, 1970s);
reimplemented from the published formula, no code borrowed. One of the most
common trend-following building blocks in public crypto bot repos.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "macd_trend",
    "category": "trend_following",
    "source": "classic concept (Gerald Appel's MACD)",
    "license": "N/A",
    "description": (
        "Long while the MACD line is above its signal line, flat (or short "
        "if `allow_short`) while below."
    ),
    "default_params": {"fast": 12, "slow": 26, "signal": 9, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    short_weight = -1.0 if allow_short else 0.0
    return (macd_line > signal_line).map({True: 1.0, False: short_weight})
