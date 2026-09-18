"""New York "Kill Zone" opening-range breakout.

Source: ICT / Smart Money Concepts methodology — the NY Kill Zone
(~8-11am New York time) is taught as the highest-institutional-volume
window of the trading day, where the session's opening range often gets
broken decisively. Uses `America/New_York` for the session window so
EST/EDT daylight-saving transitions are handled correctly (unlike this
repo's existing pattern_london_breakout.py, which uses a fixed UTC window
and silently drifts by an hour across DST changes — a known limitation
noted here rather than fixed there, to keep that module's behavior
unchanged for any results already generated with it).

Concept borrowed from forex/traditional markets where "session open" is a
real, distinct market event; crypto trades 24/7 with no equivalent close,
so treat this as a documented but more speculative hypothesis for crypto
than for FX.
License: N/A (public trading concept, reimplemented).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "smc_ny_killzone_breakout",
    "category": "smart_money_concepts",
    "source": "ICT / Smart Money Concepts, 'New York Kill Zone' opening-range breakout",
    "license": "N/A (public trading concept, reimplemented)",
    "description": (
        "Builds the day's opening range from the first `range_hours` of the "
        "NY Kill Zone window (America/New_York local time), then goes "
        "long/short on a breakout above/below that range for the rest of "
        "the zone; flat outside the window."
    ),
    "default_params": {"zone_start_hour": 8, "range_hours": 1, "zone_end_hour": 11},
}


def signals(
    df: pd.DataFrame, zone_start_hour: int = 8, range_hours: int = 1, zone_end_hour: int = 11, **_
) -> pd.Series:
    ny_time = df.index.tz_convert("America/New_York")
    ny_date = ny_time.date
    ny_hour = ny_time.hour

    in_range_window = (ny_hour >= zone_start_hour) & (ny_hour < zone_start_hour + range_hours)
    in_zone = (ny_hour >= zone_start_hour) & (ny_hour < zone_end_hour)

    df_local = pd.DataFrame({"close": df["close"].to_numpy(), "date": ny_date}, index=df.index)
    range_high = df_local["close"].where(in_range_window).groupby(df_local["date"]).transform("max")
    range_low = df_local["close"].where(in_range_window).groupby(df_local["date"]).transform("min")
    range_high = range_high.groupby(df_local["date"]).ffill()
    range_low = range_low.groupby(df_local["date"]).ffill()

    close = df["close"]
    breakout_long = in_zone & (close > range_high) & range_high.notna()
    breakout_short = in_zone & (close < range_low) & range_low.notna()

    weight = pd.Series(0.0, index=df.index)
    weight[breakout_long] = 1.0
    weight[breakout_short] = -1.0
    weight[~in_zone] = 0.0
    return weight
