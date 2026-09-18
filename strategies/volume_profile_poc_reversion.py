"""Rolling Volume Profile / Point-of-Control (POC) mean reversion.

Source: classic concept (Market/Volume Profile, developed by Peter
Steidlmayer at the CBOT; the Point of Control is the well-documented
name for the price level that traded the most volume within a session or
lookback window). This strategy rebuilds a coarse volume profile over a
rolling lookback window on every bar, finds the POC, and fades price back
toward it when the current close has drifted meaningfully away — the
standard "value area" intuition that the heaviest-traded price acts as a
magnet. Reimplemented from the publicly documented concept, no code
borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "volume_profile_poc_reversion",
    "category": "mean_reversion",
    "source": "classic concept (Market/Volume Profile Point of Control)",
    "license": "N/A",
    "description": (
        "Over each rolling `profile_window`-bar window, bins typical price "
        "into `num_bins` buckets weighted by volume and takes the bucket "
        "with the most volume as the Point of Control (POC). Long when "
        "close sits more than `dev_pct` below POC (expects reversion up to "
        "the heaviest-traded price), short (if `allow_short`) when more "
        "than `dev_pct` above, flat within the band."
    ),
    "default_params": {
        "profile_window": 48,
        "num_bins": 12,
        "dev_pct": 0.01,
        "allow_short": False,
    },
}


def _rolling_poc(typical: pd.Series, volume: pd.Series, window: int, num_bins: int) -> pd.Series:
    typ = typical.to_numpy(dtype=float)
    vol = volume.to_numpy(dtype=float)
    n = len(typ)
    poc = np.full(n, np.nan)

    for i in range(window - 1, n):
        seg_typ = typ[i - window + 1 : i + 1]
        seg_vol = vol[i - window + 1 : i + 1]
        lo, hi = seg_typ.min(), seg_typ.max()
        if hi <= lo:
            poc[i] = seg_typ[-1]
            continue
        edges = np.linspace(lo, hi, num_bins + 1)
        idx = np.clip(np.digitize(seg_typ, edges) - 1, 0, num_bins - 1)
        vol_by_bin = np.bincount(idx, weights=seg_vol, minlength=num_bins)
        top = vol_by_bin.argmax()
        poc[i] = (edges[top] + edges[top + 1]) / 2

    return pd.Series(poc, index=typical.index)


def signals(
    df: pd.DataFrame,
    profile_window: int = 48,
    num_bins: int = 12,
    dev_pct: float = 0.01,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    typical = (df["high"] + df["low"] + close) / 3

    poc = _rolling_poc(typical, df["volume"], profile_window, num_bins)
    deviation = (close - poc) / poc.replace(0, np.nan)

    weight = pd.Series(np.nan, index=df.index)
    weight[deviation.abs() <= dev_pct * 0.25] = 0.0
    weight[deviation < -dev_pct] = 1.0
    weight[deviation > dev_pct] = -1.0 if allow_short else 0.0
    return weight.ffill().fillna(0.0)
