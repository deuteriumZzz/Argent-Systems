"""RSI-filtered DCA — only fires a periodic buy when RSI is below a
threshold, skipping buys made into overbought conditions.

Source: pattern documented across freqtrade community DCA strategies that
gate scheduled/safety-order entries behind an RSI (or similar oversold)
filter — e.g. stash86/freqtrade_stuff's strat_dca.py. See:
https://github.com/stash86/freqtrade_stuff/blob/main/user_data/strategies/strat_dca.py
https://github.com/freqtrade/freqtrade-strategies
Reimplemented independently from the documented pattern; no code copied
(these repos are GPL-3.0 — only the RSI-gated-DCA idea is reused, not code).
License: N/A (concept, reimplemented independently).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "dca_rsi_filtered",
    "category": "accumulation",
    "source": "freqtrade community DCA strategies (RSI-gated entries), e.g. stash86/freqtrade_stuff",
    "license": "N/A (concept, reimplemented independently; source repos are GPL-3.0, no code copied)",
    "description": (
        "Periodic DCA buy every `interval_bars`, but each buy is deferred "
        "bar-by-bar until RSI(rsi_period) drops to or below "
        "`rsi_threshold` — only averages in on oversold dips instead of "
        "buying blindly on a fixed schedule."
    ),
    "default_params": {
        "interval_bars": 24 * 3,
        "num_buys": 20,
        "rsi_period": 14,
        "rsi_threshold": 55.0,
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
    interval_bars: int = 24 * 3,
    num_buys: int = 20,
    rsi_period: int = 14,
    rsi_threshold: float = 55.0,
    **_,
) -> pd.Series:
    rsi = _rsi(df["close"], rsi_period).to_numpy()
    n = len(df)
    step = 1.0 / num_buys

    weight = pd.Series(0.0, index=df.index)
    buys_done = 0
    next_check = 0
    while next_check < n and buys_done < num_buys:
        r = rsi[next_check]
        if np.isnan(r) or r <= rsi_threshold:
            buys_done += 1
            weight.iloc[next_check:] = min(1.0, buys_done * step)
            next_check += interval_bars
        else:
            next_check += 1  # try again next bar instead of skipping the whole interval
    return weight
