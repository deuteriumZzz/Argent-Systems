"""RSI(2) mean-reversion (Larry Connors style).

Source: classic technical-analysis concept, described in Connors/Alvarez
"Short Term Trading Strategies That Work"; reimplemented from the published
rules, no code borrowed. Extremely common building block across public
crypto bot repos.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "rsi_mean_reversion",
    "category": "mean_reversion",
    "source": "classic concept (Connors RSI2)",
    "license": "N/A",
    "description": (
        "Goes long when RSI(rsi_period) drops below `oversold`. Exits when "
        "price closes back above its SMA(sma_period)."
    ),
    "default_params": {"rsi_period": 2, "oversold": 10, "sma_period": 5},
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
    df: pd.DataFrame, rsi_period: int = 2, oversold: float = 10, sma_period: int = 5, **_
) -> pd.Series:
    close = df["close"]
    rsi = _rsi(close, rsi_period)
    sma = close.rolling(sma_period).mean()

    entries = rsi < oversold
    exits = close > sma

    weight = pd.Series(np.nan, index=df.index)
    weight[entries] = 1.0
    weight[exits] = 0.0
    return weight.ffill().fillna(0.0)
