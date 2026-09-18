"""CCI recovery from oversold confirmed by a fresh AROON-up swing.

Source: reimplemented from iterativv/NostalgiaForInfinity buy_condition_10
(conceptually adapted, no code copied) — the original rule (labelled "CCI
Divergence + AROON Trend Birth") waits for CCI to dip below -50 (but not
below -200, ruling out a full collapse) and then turn back up, while AROON
Up on a lower timeframe has just swung above 50 (a fresh short-term
upswing), gated by a higher-timeframe (4h) uptrend-or-not-oversold check.
Adapted to a single timeframe: the source's separate 1h CCI / 15m AROON /
4h trend checks all run on this dataframe's own bars, and the 4h trend
gate becomes a slower EMA slope filter.
License: GPL-3.0 (source); this reimplementation is original code
expressing the same public rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "nfi_cci_aroon_reversal",
    "category": "mean_reversion",
    "source": "reimplemented from iterativv/NostalgiaForInfinity buy_condition_10 (conceptually adapted, no code copied)",
    "license": "GPL-3.0 (source); this reimplementation is original code expressing the same public rule",
    "description": (
        "Long entry when CCI(cci_period) recovers (turns up) from between "
        "-`cci_floor` and -`cci_ceiling`, AROON-Up(aroon_period) is above "
        "`aroon_min` (a fresh upswing), and the slower trend EMA is rising "
        "(not a falling knife). Exits when CCI rises back above "
        "`cci_exit`."
    ),
    "default_params": {
        "cci_period": 20,
        "cci_ceiling": 50.0,
        "cci_floor": 200.0,
        "cci_exit": 100.0,
        "aroon_period": 14,
        "aroon_min": 50.0,
        "trend_ema_period": 100,
        "trend_slope_lookback": 10,
        "allow_short": False,
    },
}


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    typical = (high + low + close) / 3
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical - sma) / (0.015 * mad.replace(0, np.nan))


def _aroon_up(high: pd.Series, period: int) -> pd.Series:
    # bars since the highest high in the window, scaled to 0-100
    bars_since_high = high.rolling(period + 1).apply(
        lambda x: period - np.argmax(x[::-1]), raw=True
    )
    return 100 * (period - bars_since_high) / period


def _aroon_down(low: pd.Series, period: int) -> pd.Series:
    # mirror of _aroon_up: bars since the lowest low in the window
    bars_since_low = low.rolling(period + 1).apply(
        lambda x: period - np.argmin(x[::-1]), raw=True
    )
    return 100 * (period - bars_since_low) / period


def signals(
    df: pd.DataFrame,
    cci_period: int = 20,
    cci_ceiling: float = 50.0,
    cci_floor: float = 200.0,
    cci_exit: float = 100.0,
    aroon_period: int = 14,
    aroon_min: float = 50.0,
    trend_ema_period: int = 100,
    trend_slope_lookback: int = 10,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    cci = _cci(high, low, close, cci_period)
    cci_recovering = (cci > cci.shift(1)) & (cci < -cci_ceiling) & (cci > -cci_floor)

    aroon_up = _aroon_up(high, aroon_period)
    fresh_upswing = aroon_up > aroon_min

    trend_ema = close.ewm(span=trend_ema_period, adjust=False).mean()
    not_falling_knife = trend_ema > trend_ema.shift(trend_slope_lookback)

    long_entry = cci_recovering & fresh_upswing & not_falling_knife
    long_exit = cci > cci_exit

    weight = pd.Series(np.nan, index=df.index)
    # exit set first so a same-bar entry (below) can override the flatten
    weight[long_exit] = 0.0
    if allow_short:
        aroon_down = _aroon_down(low, aroon_period)
        cci_topping = (cci < cci.shift(1)) & (cci > cci_ceiling) & (cci < cci_floor)
        short_entry = cci_topping & (aroon_down > aroon_min) & (trend_ema < trend_ema.shift(trend_slope_lookback))
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)
