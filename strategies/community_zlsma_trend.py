"""ZLSMA (Zero-Lag LSMA) trend-following crossover.

Source: Pine Script community indicator (ZLSMA / "Zero Lag LSMA", widely
published on TradingView, e.g. tradingview.com/script/3LGnSrQN-ZLSMA-Zero
-Lag-LSMA by veryfid; Python port precedent at github.com/edyatl/zlsma).
Formula: fit a linear-regression moving average (LSMA) of price, fit a
second LSMA of that LSMA, then push the line forward by the gap between
the two (`zlsma = lsma + (lsma - lsma2)`) to cancel most of the lag a
plain regression line still carries. Ported to Python from the publicly
documented formula.
License: N/A (public indicator formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "community_zlsma_trend",
    "category": "trend_following",
    "source": "Pine Script community indicator (ZLSMA / Zero Lag LSMA)",
    "license": "N/A (public indicator formula)",
    "description": (
        "lsma = rolling linear-regression endpoint of close over `length` "
        "bars; lsma2 = same regression applied to lsma; "
        "zlsma = lsma + (lsma - lsma2). Long while close is above zlsma and "
        "zlsma is rising, short (if `allow_short`) while close is below a "
        "falling zlsma."
    ),
    "default_params": {"length": 32, "allow_short": False},
}


def _rolling_linreg_endpoint(series: pd.Series, length: int) -> pd.Series:
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _endpoint(y: np.ndarray) -> float:
        y_mean = y.mean()
        slope = ((x - x_mean) * (y - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        return slope * (length - 1) + intercept

    return series.rolling(length).apply(_endpoint, raw=True)


def signals(
    df: pd.DataFrame,
    length: int = 32,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]

    lsma = _rolling_linreg_endpoint(close, length)
    lsma2 = _rolling_linreg_endpoint(lsma, length)
    zlsma = lsma + (lsma - lsma2)

    zlsma_rising = zlsma > zlsma.shift(1)
    above = (close > zlsma) & zlsma_rising
    below = (close < zlsma) & ~zlsma_rising

    weight = pd.Series(np.nan, index=df.index)
    weight[above] = 1.0
    weight[below] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
