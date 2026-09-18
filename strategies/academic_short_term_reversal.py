"""Short-term reversal (Jegadeesh, 1990; Lehmann, 1990).

Source: Jegadeesh, N. (1990), "Evidence of Predictable Behavior of Security
Returns", Journal of Finance 45(3), 881-898; Lehmann, B.N. (1990), "Fads,
Martingales, and Market Efficiency", Quarterly Journal of Economics. Also
listed as a strategy page on paperswithbacktest.com / Quantpedia
(https://quantpedia.com/strategies/short-term-reversal-in-stocks/), whose
QC implementation (in paperswithbacktest/awesome-systematic-trading,
static/strategies/short-term-reversal-in-stocks.py) longs last week's
biggest losers and shorts last week's biggest winners across a stock
universe; here reimplemented for a single asset as fading its own trailing
short-horizon return (no code reused, only the documented mean-reversion
rule).
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_short_term_reversal",
    "category": "mean_reversion",
    "source": "Jegadeesh (1990) / Lehmann (1990) short-term reversal",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Fades the asset's own trailing `lookback`-bar return: sells recent "
        "winners, buys recent losers. The raw signal (negative trailing "
        "return, normalized by trailing volatility) is squashed through "
        "tanh so weight stays smoothly bounded in [-1, 1] instead of a hard "
        "clip."
    ),
    "default_params": {"lookback": 5, "vol_lookback": 20, "sensitivity": 3.0},
}


def signals(
    df: pd.DataFrame,
    lookback: int = 5,
    vol_lookback: int = 20,
    sensitivity: float = 3.0,
    **_,
) -> pd.Series:
    close = df["close"]
    trailing_return = close.pct_change(lookback)
    vol = close.pct_change().rolling(vol_lookback).std() * np.sqrt(lookback)

    normalized = (-trailing_return / vol).replace([np.inf, -np.inf], np.nan)
    weight = np.tanh(normalized * sensitivity)
    return weight.fillna(0.0)
