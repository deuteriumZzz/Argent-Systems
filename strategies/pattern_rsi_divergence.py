"""Classic price/RSI divergence.

Source: je-suis-tm/quant-trading, "RSI Pattern Recognition backtest.py"
(https://github.com/je-suis-tm/quant-trading/blob/master/RSI%20Pattern%20Recognition%20backtest.py) -
Apache License 2.0. The source's "RSI Pattern Recognition" is actually a
head-and-shoulders pattern traced directly on the RSI line (via nested
backward scans, mirroring its Bollinger Bands pattern script), not
divergence - the file's own comments explicitly dismiss RSI divergence as
unreliable and decline to implement it. Per this project's task, we
reimplement the *divergence* idea instead (classic technical-analysis
concept, not this source's code): price makes a lower low while RSI makes
a higher low (bullish), or the mirror at highs (bearish). No code copied
from the source file.
License: Apache-2.0 (source repo, for the general RSI-pattern subject
matter this module is filed under).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "pattern_rsi_divergence",
    "category": "mean_reversion",
    "source": "je-suis-tm/quant-trading (RSI Pattern Recognition backtest.py) - reimplemented as classic RSI divergence",
    "license": "Apache-2.0",
    "description": (
        "Tracks confirmed swing lows/highs in price and RSI(rsi_period) "
        "over a `pivot_window`-bar confirmation lag. Bullish divergence "
        "(price lower low, RSI higher low, within `lookback` bars of the "
        "prior swing low) goes long. Bearish divergence (price higher "
        "high, RSI lower high) goes short if `allow_short` else flat."
    ),
    "default_params": {
        "rsi_period": 14,
        "pivot_window": 3,
        "lookback": 30,
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
    pivot_window: int = 3,
    lookback: int = 30,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    rsi = _rsi(close, rsi_period)

    window = pivot_window * 2 + 1
    is_low = (close == close.rolling(window, center=True).min()).fillna(False).to_numpy()
    is_high = (close == close.rolling(window, center=True).max()).fillna(False).to_numpy()

    close_np = close.to_numpy()
    rsi_np = rsi.to_numpy()
    n = len(df)

    weight = np.zeros(n)
    position = 0.0
    short_weight = -1.0 if allow_short else 0.0
    last_low_idx, last_high_idx = -1, -1

    # `confirm_idx` lags the current bar `t` by `pivot_window` bars - that's
    # the earliest point a centered rolling min/max at that index is fully
    # known, so this stays causal (no look-ahead) despite using a centered
    # window to find the pivots themselves.
    for t in range(window - 1, n):
        confirm_idx = t - pivot_window

        if is_low[confirm_idx]:
            if (
                last_low_idx != -1
                and confirm_idx - last_low_idx <= lookback
                and close_np[confirm_idx] < close_np[last_low_idx]
                and rsi_np[confirm_idx] > rsi_np[last_low_idx]
            ):
                position = 1.0
            last_low_idx = confirm_idx

        if is_high[confirm_idx]:
            if (
                last_high_idx != -1
                and confirm_idx - last_high_idx <= lookback
                and close_np[confirm_idx] > close_np[last_high_idx]
                and rsi_np[confirm_idx] < rsi_np[last_high_idx]
            ):
                position = short_weight
            last_high_idx = confirm_idx

        weight[t] = position

    return pd.Series(weight, index=df.index)
