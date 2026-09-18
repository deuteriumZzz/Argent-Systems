"""ATR-adaptive grid — grid step widens/narrows with realized volatility.

Source: mechanic documented by Wick Hunter's ATR-Based Grid Bot ("grid step
= ATR(length) x multiplier", recomputed as volatility changes) and the
general "dynamic grid spacing with ATR" pattern described across grid-bot
docs. See:
https://docs.wickhunter.io/en/articles/11826956-atr-based-grid-bot-full-setup-guide
https://dev.to/jmolinasoler/dynamic-grid-spacing-with-atr-letting-volatility-set-the-parameters-4jo0
Reimplemented from the publicly documented mechanic, no code borrowed.
License: N/A (concept).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "grid_atr_adaptive",
    "category": "grid",
    "source": "Wick Hunter ATR-Based Grid Bot / generic ATR-grid-spacing pattern",
    "license": "N/A",
    "description": (
        "Grid step recomputed every `recalc_bars` bars as "
        "`atr_multiplier * ATR(atr_period)` instead of a fixed percentage, "
        "so the grid widens in choppy/volatile stretches and tightens when "
        "the market is quiet. Buys one unit each time price drops a full "
        "step below the last buy anchor, sells a unit once price rises a "
        "step above the price it was bought at."
    ),
    "default_params": {
        "atr_period": 14,
        "atr_multiplier": 1.0,
        "recalc_bars": 24,
        "num_grids": 10,
    },
}


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def signals(
    df: pd.DataFrame,
    atr_period: int = 14,
    atr_multiplier: float = 1.0,
    recalc_bars: int = 24,
    num_grids: int = 10,
    **_,
) -> pd.Series:
    close = df["close"].to_numpy()
    atr = _atr(df, atr_period).to_numpy()
    n = len(close)
    unit = 1.0 / num_grids

    def step_at(i: int) -> float:
        a = atr[i]
        return atr_multiplier * a if a == a and a > 0 else close[i] * 0.01 * atr_multiplier

    weight = pd.Series(0.0, index=df.index)
    open_buys: list[float] = []
    current_weight = 0.0
    anchor = close[0]
    step = step_at(0)

    for i in range(n):
        price = close[i]
        if i > 0 and i % recalc_bars == 0:
            anchor = price
            step = step_at(i)

        while anchor - price >= step and len(open_buys) < num_grids:
            anchor -= step
            open_buys.append(anchor)
            current_weight = min(1.0, current_weight + unit)

        for buy_price in list(open_buys):
            if price >= buy_price + step:
                open_buys.remove(buy_price)
                current_weight = max(0.0, current_weight - unit)

        weight.iloc[i] = current_weight

    return weight
