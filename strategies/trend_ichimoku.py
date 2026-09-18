"""Ichimoku Cloud: price vs. the Kumo cloud.

Source: classic concept (Goichi Hosoda's Ichimoku Kinko Hyo, published in
Japan in the late 1960s). The price-vs-cloud trend filter is a very common
building block in public crypto bot repos, e.g. the Obelisk_Ichimoku_* family
of freqtrade community strategies (brookmiles/freqtrade-stuff,
PeetCrypto/freqtrade-stuff). Reimplemented here from the standard published
formula only, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "trend_ichimoku",
    "category": "trend_following",
    "source": "classic concept (Goichi Hosoda's Ichimoku Kinko Hyo cloud)",
    "license": "N/A",
    "description": (
        "Long while close is above the Kumo cloud (max of Senkou Span A/B, "
        "displaced forward), flat (or short if `allow_short`) while below."
    ),
    "default_params": {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_b_period": 52,
        "displacement": 26,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]

    tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
    kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(displacement)
    span_b = (
        (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2
    ).shift(displacement)

    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)

    short_weight = -1.0 if allow_short else 0.0
    weight = pd.Series(0.0, index=df.index)
    weight[close > cloud_top] = 1.0
    weight[close < cloud_bottom] = short_weight
    return weight
