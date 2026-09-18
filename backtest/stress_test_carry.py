"""Funding-squeeze stress test for carry_static_positive_funding.

results/hedged_portfolio_summary.csv shows this leg's backtest Sharpe as
10.6 — not because it is that good, but because BTC/USDT perp funding on
Binance in this sample never printed a severe, sustained negative streak
(the script prints the actual worst single print and its date at
runtime). That is a mild number: funding on other
perps/exchanges has printed -1% to -2%/8h during real short squeezes
(shorts get forcibly unwound, paying whoever is still short). A backtest
window that never contains that event will always underprice this leg's
tail risk, no matter how much history it covers — this is a scenario
injection, not a historical replay, precisely because the real thing is
absent from our sample.

Injects a sustained severe negative-funding window (severity/duration are
CLI-adjustable, not fitted) into the real funding-rate series, recomputes
the carry leg's own returns and the capped-risk-parity portfolio's
returns from build_hedged_portfolio.py with the stressed leg substituted
in, and reports the damage — to the leg alone and to the blended
portfolio at its current 35% cap.

Usage:
    python backtest/stress_test_carry.py
    python backtest/stress_test_carry.py --shock -0.008 --days 14
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backtest.build_hedged_portfolio import CANDIDATES, MAX_LEG_WEIGHT, VOL_LOOKBACK, capped_risk_parity_weights
from backtest.portfolio_diversification_check import (
    CARRY_FLIP_COST,
    get_daily_returns,
    max_drawdown_pct,
    risk_summary,
    sharpe_stats,
)
from data.funding import fetch_funding_rate
from strategies.registry import load_strategies

START = "2020-01-01"
SHOCK_START = "2024-06-01"  # arbitrary calm-period anchor, chosen before looking at the outcome


def carry_returns_from_funding(funding_df: pd.DataFrame, strat) -> pd.Series:
    """Same PnL model as portfolio_diversification_check.get_daily_returns's
    funding_rate branch, factored out so it can be re-run on a shocked series."""
    weight = strat.signals(funding_df, **strat.meta["default_params"])
    position = weight.shift(1).fillna(0.0)
    period_return = (
        position * funding_df["funding_rate"]
        - position.diff().fillna(position).abs() * CARRY_FLIP_COST
    )
    daily = (1 + period_return).groupby(period_return.index.normalize()).prod() - 1
    daily.index = pd.to_datetime(daily.index, utc=True)
    return daily


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shock", type=float, default=-0.008, help="funding_rate per 8h during the shock window (e.g. -0.008 = -0.8%/8h)")
    ap.add_argument("--days", type=int, default=14, help="shock window length in days (~3 funding prints/day)")
    args = ap.parse_args()

    strategies = {s.meta["name"]: s for s in load_strategies()}
    carry_strat = strategies["carry_static_positive_funding"]

    funding_df = fetch_funding_rate("BTC/USDT:USDT", START)
    print(f"Real funding history: {len(funding_df)} prints, "
          f"{funding_df.index.min().date()} - {funding_df.index.max().date()}")
    print(f"Worst real print: {funding_df['funding_rate'].min():.5f} "
          f"on {funding_df['funding_rate'].idxmin().date()}")

    baseline_carry = carry_returns_from_funding(funding_df, carry_strat)
    print(f"\nBaseline carry leg (real data): sharpe={sharpe_stats(baseline_carry)[0]:.3f}, "
          f"max_dd={max_drawdown_pct(baseline_carry):.3f}%")

    # --- Inject the shock ---
    shocked = funding_df.copy()
    shock_end = pd.Timestamp(SHOCK_START, tz="UTC") + pd.Timedelta(days=args.days)
    mask = pd.Series(
        (shocked.index >= pd.Timestamp(SHOCK_START, tz="UTC")) & (shocked.index < shock_end),
        index=shocked.index,
    )
    n_prints = int(mask.sum())
    shocked.loc[mask, "funding_rate"] = args.shock
    print(f"\nInjected shock: funding_rate={args.shock:.4f}/8h for {n_prints} prints "
          f"({args.days}d starting {SHOCK_START}) — {args.shock/0.00119:.1f}x worse than the real worst print")

    stressed_carry = carry_returns_from_funding(shocked, carry_strat)
    stressed_sharpe, _, _, _ = sharpe_stats(stressed_carry)
    day_mask = mask.groupby(mask.index.normalize()).any().reindex(stressed_carry.index, fill_value=False)
    window_loss = (1 + stressed_carry[day_mask]).prod() - 1
    print(f"Stressed carry leg (whole series): sharpe={stressed_sharpe:.3f}, "
          f"max_dd={max_drawdown_pct(stressed_carry):.3f}%")
    print(f"Carry leg's own return during the {args.days}d shock window alone: {window_loss * 100:.2f}%")

    # --- Propagate into the capped-risk-parity portfolio ---
    returns = {name: get_daily_returns(name, dtype, strategies) for name, dtype in CANDIDATES}
    returns_df = pd.DataFrame(returns).dropna()

    stressed_returns_df = returns_df.copy()
    stressed_carry_aligned = stressed_carry.reindex(stressed_returns_df.index).fillna(0.0)
    stressed_returns_df["carry_static_positive_funding"] = stressed_carry_aligned

    baseline_weights = capped_risk_parity_weights(returns_df, VOL_LOOKBACK, MAX_LEG_WEIGHT)
    baseline_port = (returns_df * baseline_weights).sum(axis=1, min_count=1).dropna()

    stressed_weights = capped_risk_parity_weights(stressed_returns_df, VOL_LOOKBACK, MAX_LEG_WEIGHT)
    stressed_port = (stressed_returns_df * stressed_weights).sum(axis=1, min_count=1).dropna()

    print(f"\n=== Capped risk-parity portfolio (max {MAX_LEG_WEIGHT:.0%}/leg): baseline vs shocked ===")
    b_sharpe, _, _, _ = sharpe_stats(baseline_port)
    s_sharpe, _, _, _ = sharpe_stats(stressed_port)
    print(f"  baseline: sharpe={b_sharpe:.3f}, max_dd={max_drawdown_pct(baseline_port):.2f}%")
    print(f"  shocked:  sharpe={s_sharpe:.3f}, max_dd={max_drawdown_pct(stressed_port):.2f}%")

    def _window_mask(index: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(
            (index >= pd.Timestamp(SHOCK_START, tz="UTC")) & (index < shock_end), index=index
        )

    b_window_ret = (1 + baseline_port[_window_mask(baseline_port.index)]).prod() - 1
    s_window_ret = (1 + stressed_port[_window_mask(stressed_port.index)]).prod() - 1
    weights_window = _window_mask(baseline_weights.index)
    carry_avg_weight_in_window = baseline_weights.loc[
        weights_window, "carry_static_positive_funding"
    ].mean()
    print(f"\nPortfolio return during the {args.days}d shock window: "
          f"baseline={b_window_ret*100:.2f}% vs shocked={s_window_ret*100:.2f}% "
          f"(carry leg averaged {carry_avg_weight_in_window:.1%} of portfolio weight going into it)")

    summary = pd.DataFrame([
        risk_summary("carry_baseline", baseline_carry),
        risk_summary("carry_shocked", stressed_carry),
        risk_summary("portfolio_capped_rp_baseline", baseline_port),
        risk_summary("portfolio_capped_rp_shocked", stressed_port),
    ])
    summary.to_csv("results/carry_stress_test.csv", index=False)
    print("\nSaved to results/carry_stress_test.csv")


if __name__ == "__main__":
    main()
