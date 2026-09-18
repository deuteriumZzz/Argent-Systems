"""Explicit bear/crash hedge: short only when a downtrend is confirmed AND
realized volatility is elevated, flat otherwise. Never goes long.

Motivation, grounded directly in this project's own results: every
strategy examined so far (results/regime_comparison_robustness.csv, the
BitbotBY futures backtest, results/risk_summary.csv) shares the same
failure mode — negative Sharpe in bear_2018/bear_2022/correction_2021,
because almost everything here is a long-biased bet on BTC. None of them
fixes this by getting smarter; they just don't hold a leg that profits
when price falls. This strategy is that missing leg, kept deliberately
dumb and narrow (only two conditions, both directly tied to the regimes
it needs to cover) so it doesn't repeat custom_regime_adaptive_composite
v1's mistake of adding discretionary branches that don't survive
walk-forward (results/composite_tuning_top10_oos.csv).
License: N/A (custom idea, unvalidated — a hedge leg's job is to reduce
portfolio tail risk, not to have a good standalone Sharpe on its own; judge
it by what it does to the blended portfolio in
backtest/portfolio_diversification_check.py, not in isolation).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "hedge_trend_vol_bear",
    "category": "hedge",
    "source": "original (this project), combines a trend filter with a realized-vol filter",
    "license": "N/A (custom idea, unvalidated)",
    "description": (
        "Short when price is below its long-term SMA (confirmed downtrend) "
        "and realized volatility is elevated versus its own recent history "
        "(confirmed stress); flat otherwise. Never long — a hedge leg, not "
        "a directional bet."
    ),
    "default_params": {
        "trend_ma": 200,
        "vol_window": 14,
        "vol_lookback": 90,
        "vol_zscore_threshold": 1.0,
    },
}


def signals(
    df: pd.DataFrame,
    trend_ma: int = 200,
    vol_window: int = 14,
    vol_lookback: int = 90,
    vol_zscore_threshold: float = 1.0,
    **_,
) -> pd.Series:
    close = df["close"]
    downtrend = close < close.rolling(trend_ma).mean()

    returns = close.pct_change()
    realized_vol = returns.rolling(vol_window).std()
    vol_mean = realized_vol.rolling(vol_lookback).mean()
    vol_std = realized_vol.rolling(vol_lookback).std()
    vol_zscore = (realized_vol - vol_mean) / vol_std.replace(0, float("nan"))
    stressed = vol_zscore >= vol_zscore_threshold

    short = downtrend & stressed
    return short.astype(float).fillna(0.0) * -1.0
