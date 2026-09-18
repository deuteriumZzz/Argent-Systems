"""Coral Trend Indicator — six-pass cascaded EMA (T3-style) trend line.

Source: Pine Script community indicator (Coral Trend, popularized on
TradingView by contributors such as DaviddTech, building on Tim Tillson's
T3 moving average formula — see e.g. tradingview.com/script/bOpIeGkJ-Coral
-Trend-Indicator-DaviddTech, pineify.app/pine-script/indicators/coral-trend
-indicator). The Coral line is a T3 moving average colored by its own
slope: it cascades price through six exponential passes (alpha derived
from `period`) and recombines them with cubic-polynomial weights built
from a smoothing constant `cd`, producing a line that hugs price closer
than a plain EMA with far less lag or overshoot. Ported to Python from
the publicly documented formula.
License: N/A (public indicator formula).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "community_coral_trend",
    "category": "trend_following",
    "source": "Pine Script community indicator (Coral Trend Indicator, based on Tillson's T3 moving average)",
    "license": "N/A (public indicator formula)",
    "description": (
        "di = (period-1)/2 + 1, alpha = 2/(di+1). Six cascaded EMAs "
        "(i1..i6) of close at that alpha, recombined as "
        "coral = -cd^3*i6 + 3*(cd^2+cd^3)*i5 - 3*(2*cd^2+cd+cd^3)*i4 + "
        "(3*cd+1+cd^3+3*cd^2)*i3. Long while the coral line is rising "
        "(coral > coral.shift(1)), short (if `allow_short`) while falling."
    ),
    "default_params": {"period": 21, "cd": 0.4, "allow_short": False},
}


def signals(
    df: pd.DataFrame,
    period: int = 21,
    cd: float = 0.4,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]

    di = (period - 1) / 2 + 1
    alpha = 2 / (di + 1)

    i1 = close.ewm(alpha=alpha, adjust=False).mean()
    i2 = i1.ewm(alpha=alpha, adjust=False).mean()
    i3 = i2.ewm(alpha=alpha, adjust=False).mean()
    i4 = i3.ewm(alpha=alpha, adjust=False).mean()
    i5 = i4.ewm(alpha=alpha, adjust=False).mean()
    i6 = i5.ewm(alpha=alpha, adjust=False).mean()

    c3 = 3 * (cd**2 + cd**3)
    c4 = -3 * (2 * cd**2 + cd + cd**3)
    c5 = 3 * cd + 1 + cd**3 + 3 * cd**2

    coral = -(cd**3) * i6 + c3 * i5 + c4 * i4 + c5 * i3

    bullish = coral > coral.shift(1)
    short_weight = -1.0 if allow_short else 0.0
    return bullish.map({True: 1.0, False: short_weight})
