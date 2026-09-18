"""Coppock Curve — long-horizon momentum turn indicator.

Source: classic concept (Edwin "Sedge" Coppock, published in Barron's,
1962; documented formula as used by StockCharts/Investopedia and widely
ported as a TradingView community script). Sums two rates-of-change at
different lookbacks and smooths the sum with a weighted moving average;
originally built for monthly S&P timing, commonly adapted to shorter bars
(including crypto) by keeping the same period ratios. Reimplemented from
the publicly documented formula, no code borrowed.
License: N/A (public indicator formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "community_coppock_curve",
    "category": "trend_following",
    "source": "classic concept (Edwin Coppock's Coppock Curve)",
    "license": "N/A (public indicator formula)",
    "description": (
        "Coppock = WMA(wma_period) of [ROC(close, roc_long) + ROC(close, "
        "roc_short)], ROC in percent. Long entry when Coppock crosses "
        "above zero (long-horizon momentum turning positive), short (if "
        "`allow_short`) when it crosses below zero. Holds previous weight "
        "otherwise."
    ),
    "default_params": {
        "roc_long": 14,
        "roc_short": 11,
        "wma_period": 10,
        "allow_short": False,
    },
}


def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def signals(
    df: pd.DataFrame,
    roc_long: int = 14,
    roc_short: int = 11,
    wma_period: int = 10,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]

    roc_l = (close / close.shift(roc_long) - 1) * 100
    roc_s = (close / close.shift(roc_short) - 1) * 100
    coppock = _wma(roc_l + roc_s, wma_period)

    cross_up = (coppock > 0) & (coppock.shift(1) <= 0)
    cross_down = (coppock < 0) & (coppock.shift(1) >= 0)

    weight = pd.Series(np.nan, index=df.index)
    weight[cross_down] = -1.0 if allow_short else 0.0
    weight[cross_up] = 1.0
    return weight.ffill().fillna(0.0)
