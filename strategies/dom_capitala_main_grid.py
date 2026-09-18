"""dom-capitala007's MAIN grid layer — the core DCA-averaging-with-trailing-TP
mechanic actually deployed in that project (a private local project owned by
the same user as this repo, not a public GitHub source).

Transcribed from `backend_postgres/app/strategy/engine.py` in that project
(read directly, not guessed): ATR-adaptive averaging-down step, a trailing
take-profit that "arms" once unrealized profit clears a threshold and then
trails the post-arm price peak, and a peak-drawdown circuit breaker that
freezes new buys once account equity drawdown clears a threshold (their
spec: re-enabled only by a human, never automatically — reproduced as-is,
which means once this fires in a backtest it stays off for the rest of the
run, exactly like the real system would without operator intervention).

Faithful: FIRST_MAIN_STEP, MAIN_STEP_ATR_MULT, ATR clamp bounds, MAIN_TP_ARM/
MAIN_TP_TRAIL/MAIN_PROFIT_FLOOR, PEAK_DD_EXIT_ONLY/PEAK_DD_BEAR — all read
directly from their constants.
Approximated (their source uses per-lot Decimal bookkeeping tied to a live
ledger; this collapses to a single scalar `weight` per this repo's contract):
`max_levels` (their `LEVELS` constant wasn't visible in the code excerpt
read; defaulted to 5, a common choice for this style of grid — tune if the
real value differs), equal position size per level (their sizing has
allocation-tier and capital-guard logic this doesn't reproduce), and only
the MAIN layer — MICRO_GRID/PRE_GRID_MICRO/SCALP/BTC_CORE sub-layers and
greed-mode are not modeled.
License: Proprietary — same owner as this project; used here with
authorization, for internal comparison only, not for redistribution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "dom_capitala_main_grid",
    "category": "grid",
    "source": "dom_capitala007 (private local project, owner-authorized), backend_postgres/app/strategy/engine.py MAIN layer",
    "license": "Proprietary (same owner) — internal comparison only",
    "description": (
        "ATR-adaptive averaging-down grid: first entry on a 2% pullback from "
        "the post-flat high, subsequent entries on a 1.4x-ATR pullback from "
        "the last buy, up to max_levels. Trailing take-profit arms at +8% "
        "average unrealized profit and trails 8% off the post-arm peak, "
        "never selling below a 2% profit floor. New buys freeze permanently "
        "(no auto re-enable, matching the source's manual-reset design) once "
        "account drawdown from its own equity peak clears 20% (exit-only) or "
        "25% (bear mode)."
    ),
    "default_params": {
        "atr_period": 14,
        "min_atr_pct": 0.01,
        "max_atr_pct": 0.15,
        "first_step_pct": 0.02,
        "step_atr_mult": 1.4,
        "max_levels": 5,
        "tp_arm_pct": 0.08,
        "tp_trail_pct": 0.08,
        "profit_floor_pct": 0.02,
        "dd_exit_only_pct": 0.20,
        "dd_bear_pct": 0.25,
    },
}


def _atr_pct(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    true_range = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return atr / close


def signals(
    df: pd.DataFrame,
    atr_period: int = 14,
    min_atr_pct: float = 0.01,
    max_atr_pct: float = 0.15,
    first_step_pct: float = 0.02,
    step_atr_mult: float = 1.4,
    max_levels: int = 5,
    tp_arm_pct: float = 0.08,
    tp_trail_pct: float = 0.08,
    profit_floor_pct: float = 0.02,
    dd_exit_only_pct: float = 0.20,
    dd_bear_pct: float = 0.25,
    **_,
) -> pd.Series:
    close = df["close"].to_numpy()
    atr_pct = _atr_pct(df, atr_period).clip(min_atr_pct, max_atr_pct).to_numpy()
    returns = df["close"].pct_change().fillna(0.0).to_numpy()

    n = len(close)
    weight = np.zeros(n)

    levels_filled = 0
    entry_ref_price = close[0]
    last_buy_price = None
    buy_prices: list[float] = []
    tp_armed = False
    tp_peak_price = 0.0
    equity = 1.0
    peak_equity = 1.0
    frozen = False  # once True (exit_only/bear triggered), stays True: manual-reset-only, same as the source
    prev_weight = 0.0

    for i in range(n):
        price = close[i]

        # mark-to-market equity using yesterday's decided weight (causal)
        equity *= 1 + prev_weight * returns[i]
        if equity > peak_equity:
            peak_equity = equity
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if drawdown >= dd_bear_pct or drawdown >= dd_exit_only_pct:
            frozen = True  # PEAK_DD_BEAR implies PEAK_DD_EXIT_ONLY in the source too

        if levels_filled == 0:
            entry_ref_price = max(entry_ref_price, price) if not buy_prices else price
            if not frozen and price <= entry_ref_price * (1 - first_step_pct):
                levels_filled = 1
                buy_prices = [price]
                last_buy_price = price
        elif levels_filled < max_levels and not frozen:
            step = step_atr_mult * atr_pct[i]
            if price <= last_buy_price * (1 - step):
                levels_filled += 1
                buy_prices.append(price)
                last_buy_price = price

        if levels_filled > 0:
            avg_entry = sum(buy_prices) / len(buy_prices)
            profit_pct = (price - avg_entry) / avg_entry

            if not tp_armed and profit_pct >= tp_arm_pct:
                tp_armed = True
                tp_peak_price = price
            if tp_armed:
                tp_peak_price = max(tp_peak_price, price)
                trail_trigger = tp_peak_price * (1 - tp_trail_pct)
                if price <= trail_trigger and profit_pct >= profit_floor_pct:
                    levels_filled = 0
                    buy_prices = []
                    last_buy_price = None
                    tp_armed = False
                    tp_peak_price = 0.0
                    entry_ref_price = price

        weight[i] = levels_filled / max_levels
        prev_weight = weight[i]

    return pd.Series(weight, index=df.index)
