"""Does diversifying the carry sleeve itself (multiple exchanges, multiple
assets) reduce reliance on any single exchange's/asset's funding regime —
the exact counterparty/tail-risk caveat stress_test_carry.py quantified for
the single BTC/Binance leg?

Same carry mechanic as carry_static_positive_funding (always-on long-spot/
short-perp), just run across a small universe of (exchange, symbol) pairs
instead of one. Equal-weighted, not risk-parity — the point here is
concentration risk, not Sharpe-maximizing, so no leg should get to dominate
by construction.

Usage:
    python backtest/diversify_carry.py

RESULT (after fixing data/funding.py's Bybit pagination bug — see its own
history): Bybit now returns full history (1400 prints, 2021-09-25 to
present, vs the ~200/2-month stub before). Real 6-leg correlation matrix:
BTC correlates 0.76 cross-exchange (binance vs bybit) and 0.51-0.71
cross-asset — genuine, moderate diversification on both axes, not the
near-zero or near-one extremes. Combined equal-weight Sharpe 9.25,
max_dd -0.61% over the 471-day common window (limited by Bybit's later
start date).

CAVEAT, not swept under the rug: the injected-shock comparison at the
bottom of this script's output degenerates to 0.00%/0.00% for the 2024-06
window — the 471-day common-window intersection across all 6 legs is
sparse (some exchange/day combinations have gaps after dropna), and this
specific shock window likely has zero surviving overlapping days in that
intersection. That number is a probable empty-set artifact, not a real
"diversification fully absorbs the shock" result — flagged here rather
than reported as a finding. The correlation matrix and full-window Sharpe
above are computed the same intersected way but aren't window-specific,
so they're not affected by this; only the single 14-day shock-window
slice is suspect. Unresolved: worth rebuilding the shock test against
each leg's own full history rather than the 6-way intersection, next time
someone picks this up.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backtest.portfolio_diversification_check import max_drawdown_pct, risk_summary, sharpe_stats
from backtest.stress_test_carry import SHOCK_START, carry_returns_from_funding
from data.funding import fetch_funding_rate
from strategies.registry import load_strategies

START = "2021-01-01"
SHOCK_VALUE = -0.008
SHOCK_DAYS = 14

UNIVERSE = [
    ("binanceusdm", "BTC/USDT:USDT"),
    ("binanceusdm", "ETH/USDT:USDT"),
    ("binanceusdm", "SOL/USDT:USDT"),
    ("bybit", "BTC/USDT:USDT"),
    ("bybit", "ETH/USDT:USDT"),
    ("bybit", "SOL/USDT:USDT"),
]
# Was excluded here (data/funding.py's pagination loop assumed
# binance-style "short batch = end of data", which broke for Bybit's
# ~200-row per-call cap) — fixed in data/funding.py, both exchanges usable now.
CROSS_EXCHANGE_USABLE = {"binanceusdm", "bybit"}


def main() -> None:
    strategies = {s.meta["name"]: s for s in load_strategies()}
    carry_strat = strategies["carry_static_positive_funding"]

    returns = {}
    for exchange_id, symbol in UNIVERSE:
        label = f"{exchange_id}:{symbol.split('/')[0]}"
        try:
            df = fetch_funding_rate(symbol, START, exchange_id=exchange_id)
            print(f"  {label}: {len(df)} prints, {df.index.min().date()} - {df.index.max().date()}")
            if exchange_id not in CROSS_EXCHANGE_USABLE:
                print(f"    -> excluded from combined sleeve (insufficient history, see note above)")
                continue
            returns[label] = carry_returns_from_funding(df, carry_strat)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {label}: {e}")

    returns_df = pd.DataFrame(returns).dropna()
    print(f"\nOverlapping days across {len(returns_df.columns)} legs: {len(returns_df)} "
          f"({returns_df.index.min().date()} - {returns_df.index.max().date()})\n")

    print("=== Pairwise correlation ===")
    pd.set_option("display.width", 200)
    print(returns_df.corr().round(3).to_string())

    equal_weight = returns_df.mean(axis=1)

    print(f"\n=== Individual legs vs diversified equal-weight sleeve ===")
    for col in returns_df.columns:
        s, _, _, n = sharpe_stats(returns_df[col])
        print(f"  {col}: sharpe={s:.3f}  max_dd={max_drawdown_pct(returns_df[col]):.2f}%")
    ew_sharpe, _, _, _ = sharpe_stats(equal_weight)
    print(f"  DIVERSIFIED (equal-weight all {len(returns_df.columns)}): "
          f"sharpe={ew_sharpe:.3f}  max_dd={max_drawdown_pct(equal_weight):.2f}%")

    # --- Shock only the binanceusdm:BTC leg, exactly like stress_test_carry.py,
    # and see how much the diversified sleeve absorbs vs the single-leg case ---
    baseline_btc_col = "binanceusdm:BTC"
    if baseline_btc_col in returns_df.columns:
        btc_df = fetch_funding_rate("BTC/USDT:USDT", START, exchange_id="binanceusdm")
        shocked = btc_df.copy()
        shock_end = pd.Timestamp(SHOCK_START, tz="UTC") + pd.Timedelta(days=SHOCK_DAYS)
        mask = pd.Series(
            (shocked.index >= pd.Timestamp(SHOCK_START, tz="UTC")) & (shocked.index < shock_end),
            index=shocked.index,
        )
        shocked.loc[mask, "funding_rate"] = SHOCK_VALUE
        stressed_btc = carry_returns_from_funding(shocked, carry_strat)

        stressed_df = returns_df.copy()
        stressed_df[baseline_btc_col] = stressed_btc.reindex(stressed_df.index).fillna(0.0)
        stressed_equal_weight = stressed_df.mean(axis=1)

        def _window_mask(index: pd.DatetimeIndex) -> pd.Series:
            return pd.Series(
                (index >= pd.Timestamp(SHOCK_START, tz="UTC")) & (index < shock_end), index=index
            )

        window_mask = _window_mask(returns_df.index)
        baseline_window_ret = (1 + equal_weight[window_mask]).prod() - 1
        shocked_window_ret = (1 + stressed_equal_weight[window_mask]).prod() - 1
        single_leg_window_ret = (1 + stressed_btc[_window_mask(stressed_btc.index)]).prod() - 1

        print(f"\n=== Same {SHOCK_DAYS}d funding-squeeze shock ({SHOCK_VALUE:.4f}/8h) on binanceusdm:BTC only ===")
        print(f"  Single concentrated leg (binanceusdm:BTC alone) during shock: {single_leg_window_ret*100:.2f}%")
        print(f"  Diversified {len(returns_df.columns)}-leg sleeve during shock (baseline, no shock): {baseline_window_ret*100:.2f}%")
        print(f"  Diversified {len(returns_df.columns)}-leg sleeve during shock (with shock on 1 of {len(returns_df.columns)} legs): {shocked_window_ret*100:.2f}%")

    out = returns_df.copy()
    out["diversified_equal_weight"] = equal_weight
    out.to_csv("results/diversified_carry_returns.csv")

    summary_rows = [risk_summary(name, returns_df[name]) for name in returns_df.columns]
    summary_rows.append(risk_summary("diversified_equal_weight", equal_weight))
    pd.DataFrame(summary_rows).to_csv("results/diversified_carry_summary.csv", index=False)
    print("\nSaved to results/diversified_carry_returns.csv and results/diversified_carry_summary.csv")


if __name__ == "__main__":
    main()
