"""Fading the initial reaction to scheduled US macro data releases (NFP,
first Friday of the month, 8:30am ET; weekly Initial Jobless Claims, every
Thursday, 8:30am ET) — the mirror-image hypothesis of
academic_macro_release_momentum.py.

Source: documented market-microstructure observation that headline
reactions to scheduled macro prints often overshoot (algorithmic/headline
traders react to the number before the market has digested revisions,
context, or the full release), followed by a correction rather than
continuation. Reimplemented from the public observation, not a specific
paper. This module and academic_macro_release_momentum.py test opposite
hypotheses on the same event calendar so they can be compared directly
under this project's backtest.
License: N/A (documented macro-event effect, reimplemented from the
observation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_macro_release_fade",
    "category": "seasonality",
    "source": "Documented macro-event overreaction/fade observation (NFP / jobless claims)",
    "license": "N/A (documented macro-event effect, reimplemented from the observation)",
    "description": (
        "Same NFP/jobless-claims event calendar as "
        "academic_macro_release_momentum.py, but takes a position AGAINST "
        "the event bar's own return (fading the initial headline reaction) "
        "instead of following it, held for `hold_hours` bars."
    ),
    "default_params": {"hold_hours": 4, "nfp_weight": 1.0, "claims_weight": 0.4},
}


def signals(
    df: pd.DataFrame, hold_hours: int = 4, nfp_weight: float = 1.0, claims_weight: float = 0.4, **_
) -> pd.Series:
    et_time = df.index.tz_convert("America/New_York")
    is_release_hour = et_time.hour == 8

    is_friday = et_time.dayofweek == 4
    is_thursday = et_time.dayofweek == 3
    is_first_friday = is_friday & (et_time.day <= 7)

    nfp_hour = is_release_hour & is_first_friday
    claims_hour = is_release_hour & is_thursday

    returns = df["close"].pct_change()
    fade_direction = -np.sign(returns)

    weight = pd.Series(0.0, index=df.index)
    weight[nfp_hour] = fade_direction[nfp_hour] * nfp_weight
    weight[claims_hour & ~nfp_hour] = fade_direction[claims_hour & ~nfp_hour] * claims_weight

    weight = weight.replace(0.0, np.nan)
    weight = weight.ffill(limit=hold_hours).fillna(0.0)
    return weight
