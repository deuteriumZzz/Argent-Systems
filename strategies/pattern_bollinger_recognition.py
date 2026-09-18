"""Simplified W-bottom / M-top pattern recognition within Bollinger Bands.

Source: je-suis-tm/quant-trading, "Bollinger Bands Pattern Recognition
backtest.py"
(https://github.com/je-suis-tm/quant-trading/blob/master/Bollinger%20Bands%20Pattern%20Recognition%20backtest.py) -
Apache License 2.0. The source locates a precise 5-node W shape (touch
lower band, bounce to mid band, second touch no lower than the first,
prior peak, breakout above upper band) via four nested backward-scanning
loops over a 75-bar window - O(n^2)-ish and intricate to port faithfully.

Simplification (noted per task): this reimplements the same *idea* - two
lower-band touches of similar depth separated by a bounce above the mid
band, confirmed by a subsequent close back above the mid band - as a
single-pass O(n) state machine instead of nested backward scans. The M-top
(mirror, for exit/short) is the same state machine flipped around the
upper band. No code copied, only the documented double-bottom/top concept.
License: Apache-2.0 (source repo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "pattern_bollinger_recognition",
    "category": "mean_reversion",
    "source": "je-suis-tm/quant-trading (Bollinger Bands Pattern Recognition backtest.py)",
    "license": "Apache-2.0",
    "description": (
        "Simplified W-bottom: price touches below the lower Bollinger Band, "
        "bounces above the mid band, touches the lower band again at a "
        "similar or higher level (within `tolerance`), then closes back "
        "above the mid band -> long. Mirrored M-top (touches above the "
        "upper band twice, similar height, closes back below mid band) -> "
        "short if `allow_short` else flat. Simplified from the source's "
        "exact 5-node geometric pattern match to a single-pass state "
        "machine over band touches."
    ),
    "default_params": {
        "bb_period": 20,
        "bb_std": 2.0,
        "tolerance": 0.01,
        "max_span": 60,
        "allow_short": False,
    },
}


def signals(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    tolerance: float = 0.01,
    max_span: int = 60,
    allow_short: bool = False,
    **_,
) -> pd.Series:
    close = df["close"]
    mid = close.rolling(bb_period).mean()
    std = close.rolling(bb_period).std()
    upper = (mid + bb_std * std).to_numpy()
    lower = (mid - bb_std * std).to_numpy()
    mid = mid.to_numpy()
    close = close.to_numpy()

    n = len(df)
    weight = np.zeros(n)
    position = 0.0
    short_weight = -1.0 if allow_short else 0.0

    w_state, w_low, w_idx = 0, np.nan, -1
    m_state, m_high, m_idx = 0, np.nan, -1

    for i in range(bb_period, n):
        if w_state != 0 and i - w_idx > max_span:
            w_state = 0
        if m_state != 0 and i - m_idx > max_span:
            m_state = 0

        # W-bottom: touch lower band -> bounce above mid -> second similar
        # touch -> close back above mid confirms the long.
        if w_state == 0:
            if close[i] < lower[i]:
                w_state, w_low, w_idx = 1, close[i], i
        elif w_state == 1:
            if close[i] < w_low:
                w_low, w_idx = close[i], i
            elif close[i] > mid[i]:
                w_state = 2
        elif w_state == 2:
            if close[i] < lower[i]:
                if close[i] >= w_low * (1 - tolerance):
                    w_state = 3
                else:
                    w_state, w_low, w_idx = 1, close[i], i
        elif w_state == 3:
            if close[i] > mid[i]:
                position = 1.0
                w_state = 0

        # M-top: mirror of the above around the upper band.
        if m_state == 0:
            if close[i] > upper[i]:
                m_state, m_high, m_idx = 1, close[i], i
        elif m_state == 1:
            if close[i] > m_high:
                m_high, m_idx = close[i], i
            elif close[i] < mid[i]:
                m_state = 2
        elif m_state == 2:
            if close[i] > upper[i]:
                if close[i] <= m_high * (1 + tolerance):
                    m_state = 3
                else:
                    m_state, m_high, m_idx = 1, close[i], i
        elif m_state == 3:
            if close[i] < mid[i]:
                position = short_weight
                m_state = 0

        weight[i] = position

    return pd.Series(weight, index=df.index)
