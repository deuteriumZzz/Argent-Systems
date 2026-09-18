"""Long-horizon overreaction reversal (De Bondt & Thaler, 1985).

Source: De Bondt, W.F.M. & Thaler, R. (1985), "Does the Stock Market
Overreact?", Journal of Finance 40(3), 793-805 - finds that portfolios of
long-horizon "losers" outperform long-horizon "winners" over the subsequent
long-horizon period, attributed to systematic overreaction that later
corrects. The original paper uses 3-5 year formation/test periods on equities;
crypto's much faster information/price cycles are compressed here to a
multi-week formation window (still an order of magnitude longer than
academic_short_term_reversal.py's default 5-bar and academic_one_day_reversal.py's
1-bar horizons, keeping this a genuinely distinct published-horizon result).
No code reused, only the documented long-horizon mean-reversion rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_long_term_overreaction_reversal",
    "category": "mean_reversion",
    "source": "De Bondt & Thaler (1985), 'Does the Stock Market Overreact?', Journal of Finance 40(3)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Fades the asset's own trailing `formation_bars`-bar return (a "
        "long-horizon formation period), normalized by trailing volatility "
        "and squashed through tanh: long-horizon losers get bought, "
        "long-horizon winners get sold."
    ),
    "default_params": {"formation_bars": 24 * 14, "vol_lookback": 24 * 7, "sensitivity": 2.0},
}


def signals(
    df: pd.DataFrame,
    formation_bars: int = 24 * 14,
    vol_lookback: int = 24 * 7,
    sensitivity: float = 2.0,
    **_,
) -> pd.Series:
    close = df["close"]
    formation_return = close.pct_change(formation_bars)
    vol = close.pct_change().rolling(vol_lookback).std() * np.sqrt(formation_bars)

    normalized = (-formation_return / vol).replace([np.inf, -np.inf], np.nan)
    weight = np.tanh(normalized * sensitivity)
    return weight.fillna(0.0)
