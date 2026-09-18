"""MA crossover gated by an ADX entry/exit regime (asymmetric thresholds).

Source: jesse-ai/example-strategies, MAGen/__init__.py
(https://github.com/jesse-ai/example-strategies/blob/master/MAGen/__init__.py) —
MIT licensed. The distinguishing (and less common) mechanic vs. a plain
MA-cross-with-ADX-filter is the *asymmetric* ADX gate: entry requires ADX
above a higher `adx_entry` threshold (a fresh, strong trend), while the
position is only closed once ADX decays below a lower `adx_exit` threshold
(letting a maturing trend run) rather than exiting on the first weak
reading. Reimplemented the published rule from scratch (original is a
Jesse-framework event-driven Strategy class), no code copied.
License: MIT (source repo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "niche_ma_adx_regime",
    "category": "trend_following",
    "source": "jesse-ai/example-strategies (MAGen)",
    "license": "MIT",
    "description": (
        "Long entry when fast EMA crosses above slow EMA while ADX > "
        "`adx_entry`. Exit only once fast EMA falls back below slow EMA "
        "AND ADX has decayed below `adx_exit` (asymmetric regime gate: "
        "stricter to enter than to stay in)."
    ),
    "default_params": {
        "fast_period": 10,
        "slow_period": 30,
        "adx_period": 14,
        "adx_entry": 25,
        "adx_exit": 15,
        "allow_short": False,
    },
}


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)


def signals(
    df: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 30,
    adx_period: int = 14,
    adx_entry: float = 25,
    adx_exit: float = 15,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    fast = close.ewm(span=fast_period, adjust=False).mean()
    slow = close.ewm(span=slow_period, adjust=False).mean()
    adx = _adx(df["high"], df["low"], close, adx_period)

    cross_up = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

    long_entry = cross_up & (adx > adx_entry)
    short_entry = cross_down & (adx > adx_entry)
    long_exit = (fast < slow) & (adx < adx_exit)
    short_exit = (fast > slow) & (adx < adx_exit)

    weight = pd.Series(float("nan"), index=df.index)
    weight[long_entry] = 1.0
    if allow_short:
        weight[short_entry] = -1.0
    weight[long_exit | short_exit] = 0.0
    return weight.ffill().fillna(0.0)
