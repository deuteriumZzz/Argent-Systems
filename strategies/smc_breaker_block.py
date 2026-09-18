"""Breaker Block: a failed order block that flips role — broken support
becomes resistance (and vice versa) on the next retest.

Source: ICT / Smart Money Concepts methodology. Builds on the same
order-block detection as smc_order_block.py (reimplemented standalone
here), adding the "flip on failure" refinement ICT calls a Breaker Block.
One reasonable, backtestable interpretation, not a canonical algorithm.
License: N/A (public trading concept, reimplemented).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "smc_breaker_block",
    "category": "smart_money_concepts",
    "source": "ICT / Smart Money Concepts, 'Breaker Block'",
    "license": "N/A (public trading concept, reimplemented)",
    "description": (
        "Tracks order blocks like smc_order_block.py, but when a bullish "
        "order block (support) fails, it flips into a bearish breaker "
        "(future resistance, short on retest) instead of being discarded; "
        "mirrored for a failed bearish order block flipping into a bullish "
        "breaker (future support, long on retest)."
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

    bull_zone = None  # active bullish OB (support)
    bear_zone = None  # active bearish OB (resistance)
    bull_breaker = None  # failed bullish OB, now acts as resistance
    bear_breaker = None  # failed bearish OB, now acts as support
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
            if price < zl:
                bull_breaker = bull_zone  # support failed -> flips to resistance
                bull_zone = None
        if bear_zone is not None:
            zl, zh = bear_zone
            if price > zh:
                bear_breaker = bear_zone  # resistance failed -> flips to support
                bear_zone = None

        if bull_breaker is not None:
            zl, zh = bull_breaker
            if zl <= price <= zh:
                current_weight = -1.0
            elif price > zh:
                bull_breaker = None
                if current_weight < 0:
                    current_weight = 0.0
        if bear_breaker is not None:
            zl, zh = bear_breaker
            if zl <= price <= zh:
                current_weight = 1.0
            elif price < zl:
                bear_breaker = None
                if current_weight > 0:
                    current_weight = 0.0

        weight[i] = current_weight

    return pd.Series(weight, index=df.index)
