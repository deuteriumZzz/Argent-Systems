"""Reversal after a volume+return "event" spike - a volume-spike proxy for
post-earnings-announcement-drift-style reversal literature.

Source: Quantpedia, "Reversal during Earnings Announcements"
(https://quantpedia.com/strategies/reversal-during-earnings-announcements/);
also related to the "non-information shock" branch of Savor, P. (2012), "Stock
Returns After Major Price Shocks: The Impact of Information", Journal of
Financial Economics 106(3), 635-659, which finds large price moves NOT
accompanied by confirming news/information tend to reverse rather than drift.
Crypto has no earnings dates, so this reimplements the underlying mechanic -
"a large, abnormal one-bar price move on abnormally high volume tends to
partially revert over the following bars" - using a volume-z-score +
return-z-score event proxy in place of an earnings-announcement calendar. No
code reused, only the rule; this is an explicit proxy, not a earnings-date
replication.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_volume_spike_reversal_proxy",
    "category": "mean_reversion",
    "source": "Quantpedia 'Reversal during Earnings Announcements' / Savor (2012) JFE 106(3) non-information price-shock reversal, adapted via a volume+return z-score event proxy (no earnings data available for crypto)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Flags a bar as an 'event' when its volume z-score exceeds "
        "`volume_z_threshold` and its return z-score (both vs a trailing "
        "`vol_lookback`-bar window) exceeds `return_z_threshold` in absolute "
        "value. On an event, opens a position fading the event bar's return "
        "direction and holds it for `holding_period` bars, then flattens. "
        "Overlapping events simply overwrite the held direction."
    ),
    "default_params": {
        "volume_z_threshold": 2.5,
        "return_z_threshold": 1.5,
        "vol_lookback": 48,
        "holding_period": 6,
    },
}


def signals(
    df: pd.DataFrame,
    volume_z_threshold: float = 2.5,
    return_z_threshold: float = 1.5,
    vol_lookback: int = 48,
    holding_period: int = 6,
    **_,
) -> pd.Series:
    ret = df["close"].pct_change()

    vol_mean = df["volume"].rolling(vol_lookback).mean()
    vol_std = df["volume"].rolling(vol_lookback).std()
    volume_z = (df["volume"] - vol_mean) / vol_std

    ret_mean = ret.rolling(vol_lookback).mean()
    ret_std = ret.rolling(vol_lookback).std()
    return_z = (ret - ret_mean) / ret_std

    event = (volume_z > volume_z_threshold) & (return_z.abs() > return_z_threshold)
    direction = -np.sign(ret)

    event_positions = np.flatnonzero(event.fillna(False).to_numpy())
    w = np.zeros(len(df))
    d = direction.fillna(0.0).to_numpy()
    n = len(w)
    # ponytail: later overlapping events simply overwrite earlier ones' tail;
    # a priority queue of active events would resolve overlaps more precisely
    # if that ever matters for a real backtest.
    for i in event_positions:
        end = min(i + holding_period, n)
        w[i:end] = d[i]

    return pd.Series(w, index=df.index)
