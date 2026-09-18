"""Stochastic RSI mean reversion.

StochRSI applies the Stochastic Oscillator formula to RSI values instead of
price, producing a faster, more sensitive oversold/overbought oscillator.
Widely used in public crypto bot repos (freqtrade/jesse example strategies
routinely include it as a standard indicator) as a mean-reversion trigger.

Source: classic technical-analysis concept, developed by Tushar Chande and
Stanley Kroll (1994, "The New Technical Trader"). Reimplemented from the
published formula, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "reversion_stoch_rsi",
    "category": "mean_reversion",
    "source": "classic concept (Chande & Kroll's Stochastic RSI)",
    "license": "N/A",
    "description": (
        "RSI(rsi_period), then Stochastic formula applied to RSI over "
        "`stoch_period` bars, scaled to 0-100. Goes long when StochRSI < "
        "`oversold`, exits/shorts when StochRSI > `overbought`."
    ),
    "default_params": {
        "rsi_period": 14,
        "stoch_period": 14,
        "oversold": 20,
        "overbought": 80,
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
    rsi_period: int = 14,
    stoch_period: int = 14,
    oversold: float = 20,
    overbought: float = 80,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    rsi = _rsi(close, rsi_period)
    lowest_rsi = rsi.rolling(stoch_period).min()
    highest_rsi = rsi.rolling(stoch_period).max()
    span = (highest_rsi - lowest_rsi).replace(0, np.nan)
    stoch_rsi = 100 * (rsi - lowest_rsi) / span

    long_entries = stoch_rsi < oversold
    short_entries = stoch_rsi > overbought

    weight = pd.Series(np.nan, index=df.index)
    weight[long_entries] = 1.0
    weight[short_entries] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
