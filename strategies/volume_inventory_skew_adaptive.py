"""Inventory-skew-inspired directional sizing (adapted from Hummingbot).

Source: mechanic adapted from Hummingbot's documented "Inventory Skew"
feature for its Pure Market Making strategy
(hummingbot.org/strategies/v1-strategies/strategy-configs/inventory-skew/,
Apache-2.0). The real feature adjusts bid/ask order *sizes* so a market
maker's base/quote holdings drift back toward a target ratio, and widens
effective exposure less when realized volatility is high (inventory risk
management). This repo has no bid/ask interface — only a single directional
position weight — so this is an explicit **adaptation, not the actual
market-making mechanic**: it treats "how one-sided the recent trend has
been" as a proxy for how skewed a market maker's inventory would have
become chasing that trend, and shrinks directional conviction as that
proxy grows or as realized volatility rises above its own baseline (the
same "reduce size when inventory risk is elevated" instinct, applied to a
single directional bet instead of two-sided quotes).
License: Apache-2.0 (Hummingbot, source concept); this file is an original
reimplementation of the general idea, no code borrowed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "volume_inventory_skew_adaptive",
    "category": "trend_following",
    "source": "adapted from Hummingbot's Inventory Skew feature (concept only, not the bid/ask mechanic)",
    "license": "Apache-2.0 (Hummingbot, source concept); original reimplementation",
    "description": (
        "Trend direction = sign(EMA(fast_period) - EMA(slow_period)). "
        "'Inventory pressure' = |rolling sum of that direction over "
        "`inventory_window` bars| / `inventory_window` (near 1 when the "
        "trend has been one-sided the whole window, mimicking a market "
        "maker maximally skewed into one asset). 'Vol dampener' = "
        "baseline realized vol (rolling median over `vol_baseline_window`) "
        "divided by current realized vol, clipped to [0, 1] (shrinks size "
        "when volatility runs hot, as a market maker would widen spreads "
        "and quote less size). Final weight = direction * (1 - inventory "
        "pressure) * vol dampener — full conviction only when the trend is "
        "fresh (not yet over-skewed) and volatility is calm."
    ),
    "default_params": {
        "fast_period": 12,
        "slow_period": 48,
        "inventory_window": 20,
        "vol_window": 14,
        "vol_baseline_window": 100,
        "allow_short": True,
    },
}


def signals(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 48,
    inventory_window: int = 20,
    vol_window: int = 14,
    vol_baseline_window: int = 100,
    allow_short: bool = True,
    **_,
) -> pd.Series:
    close = df["close"]

    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    direction = np.sign(ema_fast - ema_slow).fillna(0.0)

    inventory_pressure = (
        direction.rolling(inventory_window).sum().abs() / inventory_window
    ).clip(0.0, 1.0)

    returns = close.pct_change()
    realized_vol = returns.rolling(vol_window).std()
    vol_baseline = realized_vol.rolling(vol_baseline_window).median()
    vol_dampener = (vol_baseline / realized_vol.replace(0, np.nan)).clip(0.0, 1.0).fillna(0.0)

    weight = direction * (1 - inventory_pressure) * vol_dampener
    if not allow_short:
        weight = weight.clip(lower=0.0)
    return weight.fillna(0.0).clip(-1.0, 1.0)
