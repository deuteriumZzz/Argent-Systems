"""Money Flow Index (MFI) standalone midline trend filter.

Source: classic concept (Gene Quong & Avrum Soudack's Money Flow Index,
1989; documented formula as used by StockCharts/Investopedia). MFI is
"volume-weighted RSI" — same up/down ratio logic as RSI but built from
typical-price*volume flow instead of raw price change. Unlike
`nfi_mfi_cmf_bb_accumulation.py` (which gates MFI behind a Bollinger-band
location, a CMF confirmation and a double Williams %R oversold read), this
file uses MFI on its own as a pure trend filter: money flow persistently
above the midline says buying volume dominates, below says selling volume
dominates. Reimplemented from the publicly documented formula, no code
borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "volume_mfi_trend_filter",
    "category": "trend_following",
    "source": "classic concept (Gene Quong & Avrum Soudack's Money Flow Index)",
    "license": "N/A",
    "description": (
        "MFI(period) computed from typical price * volume flow (same "
        "gain/loss ratio construction as RSI). Long while MFI stays above "
        "`bull_level`, flat (or short if `allow_short`) once it drops below "
        "`bear_level`, holds previous weight in the neutral band between — "
        "a hysteresis band around the 50 midline rather than a single "
        "threshold, to avoid flip-flopping on noise."
    ),
    "default_params": {
        "period": 14,
        "bull_level": 55.0,
        "bear_level": 45.0,
        "allow_short": False,
    },
}


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    typical = (high + low + close) / 3
    raw_flow = typical * volume
    rising = typical > typical.shift(1)
    pos_flow = raw_flow.where(rising, 0.0).rolling(period).sum()
    neg_flow = raw_flow.where(~rising, 0.0).rolling(period).sum()
    money_ratio = pos_flow / neg_flow.replace(0, np.nan)
    return 100 - (100 / (1 + money_ratio))


def signals(
    df: pd.DataFrame,
    period: int = 14,
    bull_level: float = 55.0,
    bear_level: float = 45.0,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    mfi = _mfi(high, low, close, volume, period)

    weight = pd.Series(np.nan, index=df.index)
    weight[mfi > bull_level] = 1.0
    weight[mfi < bear_level] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
