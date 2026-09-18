"""Funding-rate carry, gated by a minimum-worth-it threshold.

Source: standard basis-trade refinement — only run the cash-and-carry trade
when the funding rate clears round-trip costs (two-leg fees + basis slippage
on entry/exit), otherwise sit out. Reimplemented from the public mechanic.
License: N/A (concept).

Same funding_rate data contract as carry_static_positive_funding.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "carry_funding_threshold",
    "category": "carry",
    "data_type": "funding_rate",
    "source": "generic concept (cost-aware crypto basis trade)",
    "license": "N/A",
    "description": (
        "On when funding_rate >= entry_threshold, off when it drops below "
        "exit_threshold (hysteresis avoids flipping on every small wobble "
        "around the entry level)."
    ),
    "default_params": {"entry_threshold": 0.0002, "exit_threshold": 0.0000},
}


def signals(
    df: pd.DataFrame, entry_threshold: float = 0.0002, exit_threshold: float = 0.0000, **_
) -> pd.Series:
    funding = df["funding_rate"]
    entries = funding >= entry_threshold
    exits = funding <= exit_threshold

    weight = pd.Series(np.nan, index=df.index)
    weight[entries] = 1.0
    weight[exits] = 0.0
    return weight.ffill().fillna(0.0)
