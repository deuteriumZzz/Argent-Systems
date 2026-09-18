"""TTM Squeeze (John Carter's original rules) — squeeze fire + momentum color.

Source: Pine Script / classic-indicator community port of John Carter's
(Trade the Markets / Simpler Trading) TTM Squeeze, as documented by
StockCharts ChartSchool (chartschool.stockcharts.com/.../ttm-squeeze) and
widely reproduced across TradingView community scripts (e.g. the
LazyBear-style "Squeeze Momentum Indicator" derivations). Distinct from
`nfi_squeeze_momentum_release.py` elsewhere in this repo (which requires a
minimum bar-count of prior squeeze plus a same-bar Bollinger-band breakout):
this file follows Carter's specific rules verbatim — Bollinger Bands
(20, 2) fully inside Keltner Channels (20, 1.5x ATR) marks the squeeze,
a linear-regression momentum histogram gives direction, and the trade is
taken the bar the squeeze fires in the direction the histogram already
points. Ported to Python from the publicly documented formula.
License: N/A (public indicator formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "community_ttm_squeeze_carter",
    "category": "volatility_breakout",
    "source": "Pine Script community indicator (TTM Squeeze, John Carter / Trade the Markets)",
    "license": "N/A (public indicator formula)",
    "description": (
        "Squeeze on = Bollinger Bands (period, bb_std) fully inside Keltner "
        "Channels (period, kc_atr_mult x ATR). Momentum = linear-regression "
        "endpoint, over `period` bars, of (close - avg(donchian_mid, "
        "sma(close,period))). On the bar the squeeze fires (goes from on to "
        "off): long if momentum is positive, short (if `allow_short`) if "
        "momentum is negative. Flattens on the next momentum zero-line "
        "cross (histogram color flips from the entry side)."
    ),
    "default_params": {
        "period": 20,
        "bb_std": 2.0,
        "kc_atr_mult": 1.5,
        "allow_short": False,
    },
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
    period: int = 20,
    bb_std: float = 2.0,
    kc_atr_mult: float = 1.5,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    bb_mid = close.rolling(period).mean()
    bb_stdev = close.rolling(period).std()
    bb_upper = bb_mid + bb_std * bb_stdev
    bb_lower = bb_mid - bb_std * bb_stdev

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.rolling(period).mean()
    kc_upper = bb_mid + kc_atr_mult * atr
    kc_lower = bb_mid - kc_atr_mult * atr

    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_fired = (~squeeze_on) & squeeze_on.shift(1)

    donchian_mid = (high.rolling(period).max() + low.rolling(period).min()) / 2
    baseline = (donchian_mid + bb_mid) / 2
    momentum = _rolling_linreg_endpoint(close - baseline, period)

    long_entry = squeeze_fired & (momentum > 0)
    short_entry = squeeze_fired & (momentum < 0)
    exit_cond = (momentum * momentum.shift(1)) < 0  # momentum sign flip

    weight = pd.Series(np.nan, index=df.index)
    # exit set first so a same-bar entry (below) can override the flatten
    weight[exit_cond] = 0.0
    if allow_short:
        weight[short_entry] = -1.0
    weight[long_entry] = 1.0
    return weight.ffill().fillna(0.0)
