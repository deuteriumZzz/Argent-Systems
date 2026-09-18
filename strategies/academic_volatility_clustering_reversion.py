"""Volatility-clustering-conditioned mean reversion (GARCH-style regime persistence).

Source: Bollerslev, T. (1986), "Generalized Autoregressive Conditional
Heteroskedasticity", Journal of Econometrics 31(3), 307-327 (volatility
clusters and is persistent-but-mean-reverting); French, K.R., Schwert, G.W. &
Stambaugh, R.F. (1987), "Expected Stock Returns and Volatility", Journal of
Financial Economics 19(1), 3-29 (unexpected positive volatility shocks
associate with negative contemporaneous returns, and that volatility itself
subsequently reverts to its long-run level). Reimplemented here as a
rolling-std proxy for conditional variance (no code reused, only the
documented mechanic): a real GARCH(1,1) fit would replace the ratio-of-two-
rolling-windows proxy below.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_volatility_clustering_reversion",
    "category": "mean_reversion",
    "source": "Bollerslev (1986) GARCH / French, Schwert & Stambaugh (1987) JFE 19(1) volatility-shock/return relation",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Detects a 'volatility shock' when short-window realized vol "
        "(`short_vol_window` bars) rises more than `vol_shock_threshold`x "
        "above the long-run realized vol (`long_vol_window` bars). On a "
        "shock, fades the direction of the recent move with a magnitude "
        "scaled by how extreme the vol ratio is (tanh-squashed); flat when "
        "there is no shock. ponytail: rolling-std ratio stands in for a "
        "fitted GARCH(1,1) conditional variance - swap in arch.arch_model "
        "if a real GARCH fit is ever needed."
    ),
    "default_params": {
        "short_vol_window": 12,
        "long_vol_window": 24 * 14,
        "vol_shock_threshold": 1.3,
        "sensitivity": 1.5,
    },
}


def signals(
    df: pd.DataFrame,
    short_vol_window: int = 12,
    long_vol_window: int = 24 * 14,
    vol_shock_threshold: float = 1.3,
    sensitivity: float = 1.5,
    **_,
) -> pd.Series:
    ret = df["close"].pct_change()
    short_vol = ret.rolling(short_vol_window).std()
    long_vol = ret.rolling(long_vol_window).std()
    vol_ratio = (short_vol / long_vol).replace([np.inf, -np.inf], np.nan)

    vol_shock = vol_ratio > vol_shock_threshold
    recent_return = ret.rolling(short_vol_window).sum()

    fade_strength = np.tanh((vol_ratio - 1.0).clip(lower=0) * sensitivity)
    fade = -np.sign(recent_return) * fade_strength

    weight = fade.where(vol_shock, 0.0)
    return weight.fillna(0.0)
