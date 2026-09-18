"""Order Block, but trading it the way a discretionary ICT trader
actually would — with the context smc_order_block.py's naive mechanical
version was missing (which is why that one scored among the worst in the
whole project, avg Sharpe -3.03 over 11 regimes, ~3000+ trades from
re-triggering on every touch):

  1. Higher-timeframe bias filter: only take a bullish order block long
     while price is above a long EMA (proxy for the daily/4h trend a human
     would check first), mirrored for shorts. Skips counter-trend zones
     entirely instead of trading every one.
  2. First-touch only: once a zone has produced a trade, it's marked
     consumed and won't re-trigger on further touches — a human doesn't
     re-enter the same zone repeatedly as price chops across its border.
  3. Structure confluence: only takes the trade if the order block's
     direction agrees with the current market-structure state (price above
     the last confirmed swing low for longs, below the last confirmed
     swing high for shorts) — the "OB + BOS agree" confirmation a human
     looks for instead of trading the order block alone.

Source: ICT / Smart Money Concepts methodology, refined with the above
confluence rules (not part of any single canonical ICT text — this is a
deliberate improvement on smc_order_block.py's isolated version, testing
whether adding context helps, same spirit as this repo's other custom_*
experiments).
License: N/A (public trading concept, reimplemented).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "smc_order_block_confluence",
    "category": "smart_money_concepts",
    "source": "ICT / Smart Money Concepts, 'Order Block' + HTF/structure confluence refinement",
    "license": "N/A (public trading concept, reimplemented)",
    "description": (
        "Order block trading gated by three confluence filters: "
        "higher-timeframe trend agreement (EMA), first-touch-only per zone, "
        "and market-structure agreement (swing high/low) — a mechanical "
        "proxy for how a discretionary ICT trader adds context instead of "
        "trading every zone touch in isolation."
    ),
    "default_params": {
        "impulse_lookback": 10,
        "impulse_atr_mult": 1.5,
        "atr_period": 14,
        "htf_ema_period": 100,
        "swing_window": 5,
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
    htf_ema_period: int = 100,
    swing_window: int = 5,
    **_,
) -> pd.Series:
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = _atr(df, atr_period).to_numpy()
    htf_ema = df["close"].ewm(span=htf_ema_period, adjust=False).mean().to_numpy()
    n = len(close)
    weight = np.zeros(n)
    w = swing_window

    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        if high[i] == high[i - w : i + w + 1].max():
            is_swing_high[i] = True
        if low[i] == low[i - w : i + w + 1].min():
            is_swing_low[i] = True

    bull_zone = None  # (low, high, consumed)
    bear_zone = None
    last_swing_high = None
    last_swing_low = None
    current_weight = 0.0

    for i in range(impulse_lookback, n):
        confirm_idx = i - w
        if confirm_idx >= 0:
            if is_swing_high[confirm_idx]:
                last_swing_high = high[confirm_idx]
            if is_swing_low[confirm_idx]:
                last_swing_low = low[confirm_idx]

        impulse = close[i] - close[i - impulse_lookback]
        if impulse >= impulse_atr_mult * atr[i]:
            for j in range(i - 1, i - impulse_lookback - 1, -1):
                if close[j] < open_[j]:
                    bull_zone = [low[j], high[j], False]
                    break
        elif -impulse >= impulse_atr_mult * atr[i]:
            for j in range(i - 1, i - impulse_lookback - 1, -1):
                if close[j] > open_[j]:
                    bear_zone = [low[j], high[j], False]
                    break

        price = close[i]
        htf_bullish = price > htf_ema[i]
        structure_bullish = last_swing_low is not None and price > last_swing_low

        if bull_zone is not None:
            zl, zh, consumed = bull_zone
            if price < zl:
                bull_zone = None
                if current_weight > 0:
                    current_weight = 0.0
            elif not consumed and zl <= price <= zh and htf_bullish and structure_bullish:
                current_weight = 1.0
                bull_zone[2] = True
            elif consumed and price > zh:
                if current_weight > 0:
                    current_weight = 0.0

        if bear_zone is not None:
            zl, zh, consumed = bear_zone
            structure_bearish = last_swing_high is not None and price < last_swing_high
            if price > zh:
                bear_zone = None
                if current_weight < 0:
                    current_weight = 0.0
            elif not consumed and zl <= price <= zh and not htf_bullish and structure_bearish:
                current_weight = -1.0
                bear_zone[2] = True
            elif consumed and price < zl:
                if current_weight < 0:
                    current_weight = 0.0

        weight[i] = current_weight

    return pd.Series(weight, index=df.index)
