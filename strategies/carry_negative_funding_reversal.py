"""Reversal carry: get paid to be long the perp during panic sell-offs.

Source: documented event pattern — during sharp crypto sell-offs, retail
short-sellers crowd the perp, pushing funding deeply negative (shorts pay
longs). Flipping to short-spot/long-perp during these episodes is a known
"get paid to hold the bag" trade discussed in crypto-derivatives commentary
around major crash events. Reimplemented from the public observation.
License: N/A (concept).

Same funding_rate data contract as carry_static_positive_funding.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "carry_negative_funding_reversal",
    "category": "carry",
    "data_type": "funding_rate",
    "source": "generic concept (crowded-short/panic negative-funding reversal)",
    "license": "N/A",
    "description": (
        "Flips on the short-spot/long-perp side (negative weight) when "
        "funding_rate drops below `panic_threshold` (deeply negative = "
        "crowded shorts paying longs), exits once it recovers above "
        "`exit_threshold`."
    ),
    "default_params": {"panic_threshold": -0.0005, "exit_threshold": -0.0001},
}


def signals(
    df: pd.DataFrame,
    panic_threshold: float = -0.0005,
    exit_threshold: float = -0.0001,
    **_,
) -> pd.Series:
    funding = df["funding_rate"]
    entries = funding <= panic_threshold
    exits = funding >= exit_threshold

    weight = pd.Series(np.nan, index=df.index)
    weight[entries] = -1.0
    weight[exits] = 0.0
    return weight.ffill().fillna(0.0)
