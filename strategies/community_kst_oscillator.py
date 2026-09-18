"""Know Sure Thing (KST) oscillator crossover.

Source: classic concept (Martin Pring's "Know Sure Thing", published in
Stocks & Commodities magazine, 1992; documented formula as used by
StockCharts/Investopedia and reproduced as a community indicator across
TradingView/ProRealTime). A "summed rate of change" oscillator: four ROC
lookbacks are each smoothed with their own SMA, then combined with
increasing weight given to the longer-horizon terms. Reimplemented from
the publicly documented formula, no code borrowed.
License: N/A (public indicator formula).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "community_kst_oscillator",
    "category": "trend_following",
    "source": "classic concept (Martin Pring's Know Sure Thing oscillator)",
    "license": "N/A (public indicator formula)",
    "description": (
        "RCMA1..4 = SMA(ROC(close, roc_n), sma_n) for the four (roc, sma) "
        "period pairs; KST = 1*RCMA1 + 2*RCMA2 + 3*RCMA3 + 4*RCMA4; signal "
        "= SMA(KST, signal_period). Long while KST > signal, short (if "
        "`allow_short`) while below."
    ),
    "default_params": {
        "roc1": 10, "sma1": 10,
        "roc2": 15, "sma2": 10,
        "roc3": 20, "sma3": 10,
        "roc4": 30, "sma4": 15,
        "signal_period": 9,
        "allow_short": False,
    },
}


def _roc(close: pd.Series, period: int) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def signals(
    df: pd.DataFrame,
    roc1: int = 10, sma1: int = 10,
    roc2: int = 15, sma2: int = 10,
    roc3: int = 20, sma3: int = 10,
    roc4: int = 30, sma4: int = 15,
    signal_period: int = 9,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]

    rcma1 = _roc(close, roc1).rolling(sma1).mean()
    rcma2 = _roc(close, roc2).rolling(sma2).mean()
    rcma3 = _roc(close, roc3).rolling(sma3).mean()
    rcma4 = _roc(close, roc4).rolling(sma4).mean()

    kst = rcma1 + 2 * rcma2 + 3 * rcma3 + 4 * rcma4
    signal_line = kst.rolling(signal_period).mean()

    bullish = kst > signal_line
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})
