"""Ehlers Fisher Transform reversal, gated by an EMA baseline trend filter.

Source: jaredrsommer/freqtradestrategies, GKD_FisherTransform.py
(https://github.com/jaredrsommer/freqtradestrategies/blob/master/GKD_FisherTransform.py) —
repo has no LICENSE file, so treated as license-unclear: reimplemented only
the documented rule (Fisher Transform of min-max-normalized median price,
smoothed, crossed against an EMA "baseline" trend filter), no code copied.
The Fisher Transform itself is John Ehlers' published indicator (classic
concept); this repo's specific contribution reused here is the pairing of a
smoothed Fisher crossover with a slow EMA slope as a trend gate.
License: reimplemented from public repo, source license unclear - logic only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "niche_fisher_transform",
    "category": "mean_reversion",
    "source": "jaredrsommer/freqtradestrategies (GKD_FisherTransform) + Ehlers Fisher Transform",
    "license": "reimplemented from public repo, source license unclear - logic only",
    "description": (
        "Fisher-transforms the min-max-normalized (high+low)/2 over "
        "`fisher_period` bars, smooths it with an EMA. Goes long when the "
        "smoothed Fisher line crosses above its own prior value while the "
        "EMA(baseline_period) baseline is rising; exits when Fisher turns "
        "down through zero."
    ),
    "default_params": {"fisher_period": 12, "smooth_period": 8, "baseline_period": 21},
}


def _fisher(median_price: pd.Series, period: int) -> pd.Series:
    lo = median_price.rolling(period).min()
    hi = median_price.rolling(period).max()
    norm = (2 * (median_price - lo) / (hi - lo).replace(0, np.nan) - 1).clip(-0.999, 0.999)
    return (0.5 * np.log((1 + norm) / (1 - norm))).fillna(0.0)


def signals(
    df: pd.DataFrame,
    fisher_period: int = 12,
    smooth_period: int = 8,
    baseline_period: int = 21,
    **_,
) -> pd.Series:
    median_price = (df["high"] + df["low"]) / 2
    fisher = _fisher(median_price, fisher_period)
    fisher_smooth = fisher.ewm(span=smooth_period, adjust=False).mean()

    baseline = df["close"].ewm(span=baseline_period, adjust=False).mean()
    baseline_up = baseline.diff() > 0

    entries = (fisher_smooth > fisher_smooth.shift(1)) & (fisher_smooth < 0) & baseline_up
    exits = (fisher_smooth.shift(1) > 0) & (fisher_smooth < 0)

    weight = pd.Series(float("nan"), index=df.index)
    weight[entries] = 1.0
    weight[exits] = 0.0
    return weight.ffill().fillna(0.0)
