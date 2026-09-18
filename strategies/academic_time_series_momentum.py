"""Time Series Momentum (Moskowitz, Ooi & Pedersen, 2012).

Source: Moskowitz, T.J., Ooi, Y.H., Pedersen, L.H. (2012), "Time Series
Momentum", Journal of Financial Economics 104(2), 228-250
(https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2089463). Also mirrored
as a strategy page on paperswithbacktest.com / Quantpedia
(https://quantpedia.com/strategies/time-series-momentum-effect/), which
pwb-toolbox's sibling repo paperswithbacktest/awesome-systematic-trading
lists in static/strategies/time-series-momentum-effect.py (QuantConnect
algo, MIT-licensed repo) — no code from that file is reused, only the
documented rule: sign of trailing total return determines long/short, sized
inversely to realized volatility.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_time_series_momentum",
    "category": "trend_following",
    "source": "Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum', JFE 104(2)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Goes long when the trailing `lookback`-bar return is positive and "
        "short (or flat) when negative - the paper's own-past-return signal, "
        "not cross-sectional relative strength. Position size is scaled to a "
        "`target_vol` using trailing realized volatility, same as the "
        "paper's inverse-vol weighting (their GARCH estimate is replaced "
        "here with simple rolling std, as the paper's own robustness check "
        "notes is a fine substitute)."
    ),
    "default_params": {
        "lookback": 180,
        "vol_lookback": 20,
        "target_vol": 0.02,
        "max_leverage": 1.0,
    },
}


def signals(
    df: pd.DataFrame,
    lookback: int = 180,
    vol_lookback: int = 20,
    target_vol: float = 0.02,
    max_leverage: float = 1.0,
    **_,
) -> pd.Series:
    close = df["close"]
    trailing_return = close.pct_change(lookback)
    bar_return = close.pct_change()
    realized_vol = bar_return.rolling(vol_lookback).std()

    direction = np.sign(trailing_return)
    size = (target_vol / realized_vol).replace([np.inf, -np.inf], np.nan).clip(upper=max_leverage)

    weight = (direction * size).clip(-1.0, 1.0)
    return weight.fillna(0.0)
