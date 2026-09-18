"""Opening-range breakout, adapted from FX session hours to a fixed UTC window.

Source: je-suis-tm/quant-trading, "London Breakout backtest.py"
(https://github.com/je-suis-tm/quant-trading/blob/master/London%20Breakout%20backtest.py) -
Apache License 2.0. The source trades one FX session per calendar day: the
last hour of the Tokyo session sets the day's high/low range, the first 30
minutes of the London session trade the breakout of that range (with a
"too far past the threshold" risk filter and a fixed stop/target), and the
New York close flattens everything. Crypto has no session structure - it
trades 24/7 - so this reimplementation keeps only the core mechanic (a
range that forms in an early window, then a breakout of that range trades
the rest of the day) and swaps the FX wall-clock sessions for a
configurable UTC "session" window. No code copied, only the documented
range-then-breakout idea.
License: Apache-2.0 (source repo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "pattern_london_breakout",
    "category": "trend_following",
    "source": "je-suis-tm/quant-trading (London Breakout backtest.py)",
    "license": "Apache-2.0",
    "description": (
        "Each UTC day, the first `session_hours` hours set the day's range "
        "(high/low of that window). Once the session window closes, a "
        "close above the range high goes long and a close below the range "
        "low goes short (flat if `allow_short` is False); the position "
        "holds until the opposite breakout or the next day's session "
        "resets the range."
    ),
    "default_params": {"session_hours": 4, "allow_short": False},
}


def signals(
    df: pd.DataFrame, session_hours: int = 4, allow_short: bool = False, **_
) -> pd.Series:
    day = df.index.floor("D")
    in_session = df.index.hour < session_hours

    session_high = df["high"].where(in_session).groupby(day).transform("max")
    session_low = df["low"].where(in_session).groupby(day).transform("min")

    long_break = ~in_session & (df["close"] > session_high)
    short_break = ~in_session & (df["close"] < session_low)

    weight = pd.Series(np.nan, index=df.index)
    weight[in_session] = 0.0
    weight[long_break] = 1.0
    weight[short_break] = -1.0 if allow_short else 0.0
    return weight.groupby(day).ffill().fillna(0.0)
