"""Order Block, detected on a higher timeframe (daily) and traded on the
bar-level timeframe of `df` — the actual reason smc_order_block.py and
smc_order_block_confluence.py both failed badly (avg Sharpe -3.03 and
-2.31 over 11 regimes, thousands of trades): the zone-detection threshold
(1.5x ATR over 10 bars of whatever timeframe df happens to be) was too
lenient for hourly crypto data and flagged routine volatility as
"institutional" impulses almost constantly. A discretionary ICT trader
identifies order blocks on a higher timeframe (daily/4h) and only refines
the entry on a lower one — they aren't looking at every 10-hour window on
an hourly chart.

This resamples `df` to daily bars internally, detects zones there with a
much stricter threshold (3x ATR over 5 daily bars — a genuinely large,
rare move), then projects the most recently *fully closed* daily zone
(shifted by one day, so no intraday bar ever sees a same-day zone before
that day's candle has closed) onto `df`'s own index. Keeps the first-touch
and structure-confluence filters from smc_order_block_confluence.py, since
those did help a little even though they weren't the main problem.

Source: ICT / Smart Money Concepts methodology, refined further per the
diagnosis above (not part of any single canonical ICT text).
License: N/A (public trading concept, reimplemented).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "smc_order_block_htf",
    "category": "smart_money_concepts",
    "source": "ICT / Smart Money Concepts, 'Order Block' + higher-timeframe zone detection",
    "license": "N/A (public trading concept, reimplemented)",
    "description": (
        "Order blocks detected on daily bars (3x-ATR impulse over 5 days) "
        "and traded on df's own timeframe, gated by first-touch-only and "
        "market-structure agreement — testing whether the earlier "
        "confluence version's failure was the entry filter or the "
        "(too-permissive, same-timeframe) zone detection itself."
    ),
    "default_params": {
        "htf_impulse_lookback": 5,
        "htf_impulse_atr_mult": 3.0,
        "htf_atr_period": 14,
        "swing_window": 5,
    },
}


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _daily_zones(df: pd.DataFrame, lookback: int, atr_mult: float, atr_period: int) -> pd.DataFrame:
    daily = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    d_open = daily["open"].to_numpy()
    d_high = daily["high"].to_numpy()
    d_low = daily["low"].to_numpy()
    d_close = daily["close"].to_numpy()
    d_atr = _atr(daily, atr_period).to_numpy()
    nd = len(daily)

    bull_lo = np.full(nd, np.nan)
    bull_hi = np.full(nd, np.nan)
    bear_lo = np.full(nd, np.nan)
    bear_hi = np.full(nd, np.nan)
    cur_bull, cur_bear = None, None

    for i in range(lookback, nd):
        impulse = d_close[i] - d_close[i - lookback]
        if impulse >= atr_mult * d_atr[i]:
            for j in range(i - 1, i - lookback - 1, -1):
                if d_close[j] < d_open[j]:
                    cur_bull = (d_low[j], d_high[j])
                    break
        elif -impulse >= atr_mult * d_atr[i]:
            for j in range(i - 1, i - lookback - 1, -1):
                if d_close[j] > d_open[j]:
                    cur_bear = (d_low[j], d_high[j])
                    break
        if cur_bull:
            bull_lo[i], bull_hi[i] = cur_bull
        if cur_bear:
            bear_lo[i], bear_hi[i] = cur_bear

    return pd.DataFrame(
        {"bull_lo": bull_lo, "bull_hi": bull_hi, "bear_lo": bear_lo, "bear_hi": bear_hi},
        index=daily.index,
    )


def signals(
    df: pd.DataFrame,
    htf_impulse_lookback: int = 5,
    htf_impulse_atr_mult: float = 3.0,
    htf_atr_period: int = 14,
    swing_window: int = 5,
    **_,
) -> pd.Series:
    zones = _daily_zones(df, htf_impulse_lookback, htf_impulse_atr_mult, htf_atr_period)
    # shift 1 day: an intraday bar may only see a daily zone confirmed by the END of the PREVIOUS day
    zones = zones.shift(1).reindex(df.index, method="ffill")

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(close)
    w = swing_window

    is_swing_high = np.zeros(n, dtype=bool)
    is_swing_low = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        if high[i] == high[i - w : i + w + 1].max():
            is_swing_high[i] = True
        if low[i] == low[i - w : i + w + 1].min():
            is_swing_low[i] = True

    bull_lo = zones["bull_lo"].to_numpy()
    bull_hi = zones["bull_hi"].to_numpy()
    bear_lo = zones["bear_lo"].to_numpy()
    bear_hi = zones["bear_hi"].to_numpy()

    last_swing_high, last_swing_low = None, None
    bull_consumed_key, bear_consumed_key = None, None  # remember which zone (by bounds) was already traded
    current_weight = 0.0
    weight = np.zeros(n)

    for i in range(n):
        confirm_idx = i - w
        if confirm_idx >= 0:
            if is_swing_high[confirm_idx]:
                last_swing_high = high[confirm_idx]
            if is_swing_low[confirm_idx]:
                last_swing_low = low[confirm_idx]

        price = close[i]

        if not np.isnan(bull_lo[i]):
            zl, zh = bull_lo[i], bull_hi[i]
            key = (zl, zh)
            structure_bullish = last_swing_low is not None and price > last_swing_low
            if price < zl and current_weight > 0:
                current_weight = 0.0
            elif key != bull_consumed_key and zl <= price <= zh and structure_bullish:
                current_weight = 1.0
                bull_consumed_key = key

        if not np.isnan(bear_lo[i]):
            zl, zh = bear_lo[i], bear_hi[i]
            key = (zl, zh)
            structure_bearish = last_swing_high is not None and price < last_swing_high
            if price > zh and current_weight < 0:
                current_weight = 0.0
            elif key != bear_consumed_key and zl <= price <= zh and structure_bearish:
                current_weight = -1.0
                bear_consumed_key = key

        weight[i] = current_weight

    return pd.Series(weight, index=df.index)
