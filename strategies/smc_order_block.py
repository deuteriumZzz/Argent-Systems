"""Order Block: the last opposite-direction candle before a strong
impulsive move, treated as a zone institutions are likely to defend on
retest.

Source: ICT / Smart Money Concepts (SMC) methodology (Michael J.
Huddleston, "Inner Circle Trader", popularized ~2011+). No official
formalized spec exists — this is one reasonable, backtestable
interpretation of the commonly taught rule, not a canonical algorithm;
different SMC educators define the exact impulse/mitigation rules
differently.
License: N/A (public trading concept, reimplemented from the commonly
taught rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "smc_order_block",
    "category": "smart_money_concepts",
    "source": "ICT / Smart Money Concepts, 'Order Block' concept",
    "license": "N/A (public trading concept, reimplemented)",
    "description": (
        "Marks the last bearish candle before a strong bullish impulse "
        "(bullish order block, support) or the last bullish candle before "
        "a strong bearish impulse (bearish order block, resistance). Goes "
        "long/short when price retraces back into the most recent "
        "unmitigated zone, flattens when price closes fully through it."
    ),
    "default_params": {
        "impulse_lookback": 10,
        "impulse_atr_mult": 1.5,
        "atr_period": 14,
    },
}


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def signals(
    df: pd.DataFrame,
    impulse_lookback: int = 10,
    impulse_atr_mult: float = 1.5,
    atr_period: int = 14,
    **_,
) -> pd.Series:
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = _atr(df, atr_period).to_numpy()
    n = len(close)
    weight = np.zeros(n)

    bull_zone = None
    bear_zone = None
    current_weight = 0.0

    for i in range(impulse_lookback, n):
        impulse = close[i] - close[i - impulse_lookback]
        if impulse >= impulse_atr_mult * atr[i]:
            for j in range(i - 1, i - impulse_lookback - 1, -1):
                if close[j] < open_[j]:
                    bull_zone = (low[j], high[j])
                    break
        elif -impulse >= impulse_atr_mult * atr[i]:
            for j in range(i - 1, i - impulse_lookback - 1, -1):
                if close[j] > open_[j]:
                    bear_zone = (low[j], high[j])
                    break

        price = close[i]
        if bull_zone is not None:
            zl, zh = bull_zone
            if zl <= price <= zh:
                current_weight = 1.0
            elif price < zl:
                bull_zone = None
                if current_weight > 0:
                    current_weight = 0.0
        if bear_zone is not None:
            zl, zh = bear_zone
            if zl <= price <= zh:
                current_weight = -1.0
            elif price > zh:
                bear_zone = None
                if current_weight < 0:
                    current_weight = 0.0

        weight[i] = current_weight

    return pd.Series(weight, index=df.index)
