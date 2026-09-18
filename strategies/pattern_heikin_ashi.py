"""Heikin-Ashi bullish/bearish trend filter.

Source: je-suis-tm/quant-trading, "Heikin-Ashi backtest.py"
(https://github.com/je-suis-tm/quant-trading/blob/master/Heikin-Ashi%20backtest.py) -
Apache License 2.0. The source script trades a discrete marubozu-style
trigger (a bearish HA candle with no upper wick, larger body than the prior
bar, itself following another bearish bar) plus a hard cap on stacked long
positions. This reimplementation keeps only the documented Heikin-Ashi
*transform* (recursive HA-open, HA-close = average OHLC, HA-high/low from
OHLC + HA-open/close) and applies the simpler continuous trend-filter
reading requested for this project: long while the HA candle is a clean
bullish bar (HA-close > HA-open, no lower wick), flat (or short if
`allow_short`) while it is a clean bearish bar. No code copied from the
source file - only the published HA transform formula and general
directional idea are reused.
License: Apache-2.0 (source repo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "pattern_heikin_ashi",
    "category": "trend_following",
    "source": "je-suis-tm/quant-trading (Heikin-Ashi backtest.py)",
    "license": "Apache-2.0",
    "description": (
        "Transforms OHLC into Heikin-Ashi candles (HA-close = avg of "
        "O/H/L/C, HA-open = running average of prior HA-open/HA-close, "
        "HA-high/low from OHLC + HA-open/close). Long while the HA candle "
        "is bullish with no lower wick (HA-close > HA-open and HA-open == "
        "HA-low); short (if `allow_short`, else flat) while it's a clean "
        "bearish bar (HA-close < HA-open and HA-open == HA-high); flat "
        "otherwise."
    ),
    "default_params": {"allow_short": False},
}


def _heikin_ashi(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)

    ha_close = (o + h + l + c) / 4
    ha_open = np.empty(len(df))
    ha_open[0] = o[0]
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

    ha_high = np.maximum(np.maximum(ha_open, ha_close), h)
    ha_low = np.minimum(np.minimum(ha_open, ha_close), l)
    return ha_open, ha_high, ha_low, ha_close


def signals(df: pd.DataFrame, allow_short: bool = False, **_) -> pd.Series:
    ha_open, ha_high, ha_low, ha_close = _heikin_ashi(df)

    bullish_no_lower_wick = (ha_close > ha_open) & np.isclose(ha_open, ha_low)
    bearish_no_upper_wick = (ha_close < ha_open) & np.isclose(ha_open, ha_high)

    weight = np.zeros(len(df))
    weight[bullish_no_lower_wick] = 1.0
    weight[bearish_no_upper_wick] = -1.0 if allow_short else 0.0
    return pd.Series(weight, index=df.index)
