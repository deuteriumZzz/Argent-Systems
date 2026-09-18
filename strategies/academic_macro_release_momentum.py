"""Momentum following the initial reaction to scheduled US macro data
releases: Non-Farm Payrolls (NFP, monthly, first Friday of the month,
8:30am ET) and weekly Initial Jobless Claims (every Thursday, 8:30am ET).

Source: documented market-microstructure observation that risk assets
(BTC included, via its correlation to broad risk sentiment/dollar
strength) tend to continue in the direction of their initial reaction to
major scheduled macro prints for a short window afterward, rather than
immediately mean-revert. NFP is one of the most closely watched US
macro releases globally. Reimplemented from the public observation, not
from a specific paper.
License: N/A (documented macro-event effect, reimplemented from the
public observation).

Note: this is the mirror-image use case of a macro calendar — some
systems (e.g. this project's sibling BitBotBY, which the same user also
maintains) use the NFP/jobless-claims calendar as a *blackout* (stop
trading before/through the release to avoid unpredictable volatility)
rather than a signal to trade. Both are legitimate; this module tests the
"trade the reaction" hypothesis specifically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_macro_release_momentum",
    "category": "seasonality",
    "source": "Documented macro-event momentum observation (NFP / jobless claims reaction)",
    "license": "N/A (documented macro-event effect, reimplemented from the observation)",
    "description": (
        "On the NFP hour bar (first Friday of the month, 8am ET) and the "
        "weekly jobless-claims hour bar (every Thursday, 8am ET), takes a "
        "position in the direction of that bar's own return and holds it "
        "for `hold_hours` bars. NFP gets full weight (bigger, rarer "
        "release); jobless claims gets a smaller weight (weekly, smaller "
        "impact)."
    ),
    "default_params": {"hold_hours": 4, "nfp_weight": 1.0, "claims_weight": 0.4},
}


def signals(
    df: pd.DataFrame, hold_hours: int = 4, nfp_weight: float = 1.0, claims_weight: float = 0.4, **_
) -> pd.Series:
    et_time = df.index.tz_convert("America/New_York")
    is_release_hour = et_time.hour == 8

    is_friday = et_time.dayofweek == 4  # Monday=0 ... Friday=4
    is_thursday = et_time.dayofweek == 3
    is_first_friday = is_friday & (et_time.day <= 7)

    nfp_hour = is_release_hour & is_first_friday
    claims_hour = is_release_hour & is_thursday

    returns = df["close"].pct_change()
    event_direction = np.sign(returns)

    weight = pd.Series(0.0, index=df.index)
    weight[nfp_hour] = event_direction[nfp_hour] * nfp_weight
    weight[claims_hour & ~nfp_hour] = event_direction[claims_hour & ~nfp_hour] * claims_weight

    # hold the position for hold_hours bars after the event bar, then flatten
    weight = weight.replace(0.0, np.nan)
    weight = weight.ffill(limit=hold_hours).fillna(0.0)
    return weight
