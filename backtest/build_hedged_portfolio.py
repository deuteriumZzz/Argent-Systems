"""Extends portfolio_diversification_check.py with the one leg missing from
every strategy tested in this project so far: something that actually
profits when BTC falls, instead of just going flat/losing less. Also caps
risk-parity's weight per leg, because the uncapped version in
portfolio_diversification_check.py lets carry_static_positive_funding eat
the whole portfolio (results/risk_summary.csv: risk-parity weight is
almost entirely carry) — great backtest Sharpe, but carry's low volatility
in this sample is exactly the "steady until it isn't" shape that blows up
in a funding squeeze the backtest window never saw. Capping forces the
portfolio to actually hold the other legs, at the cost of some in-sample
Sharpe.

Walk-forward split matches tune_composite.py's TRAIN_END, so this result
is comparable to that script's warning about in-sample results not
generalizing.

Usage:
    python backtest/build_hedged_portfolio.py

UPDATE (7th leg added — vol_risk_premium, backtest/run_vrp.py): correlates
weakly with every other leg (-0.06 to +0.10), genuinely diversifying.
Full-period capped-RP Sharpe actually DROPPED to 0.672 (was 0.949 with 6
legs) while max_dd improved to -23.1% (was -41.2%) — a real Sharpe/
drawdown tradeoff, not a strict improvement either way.

MORE IMPORTANT, and NOT just a reassuring number this time: walk-forward
train (2021-2022) Sharpe is -0.774, PSR=0.0000 -- decisively bad, not
noise -- while test (2023-2025) is +1.634. Unlike v1 composite's failure
mode (great in-sample, collapses out-of-sample from overfit parameters),
this is the mirror image: bad in-sample, great out-of-sample, with no
fitted parameters to blame. The honest read is NOT "this generalizes
well" -- it's that 2021-2022 contains the worst crash cluster in the
whole sample (Terra/Luna, Celsius/3AC, FTX) and 2023-2025 was a broad
recovery/bull period; the split point happens to divide "the hard regime"
from "the easy one" almost cleanly. A portfolio's OOS Sharpe looking great
right after its train period contains the worst crashes on record is
exactly the kind of result that needs a second, differently-placed split
before being trusted -- not evidence of robustness on its own.

FOLLOW-UP, assumption-free per-year breakdown (no train/test framing at
all, just sharpe_stats per calendar year on the saved capped-RP column in
results/hedged_portfolio_returns.csv):
  2021: -0.84   2022: -0.72   2023: +2.57   2024: +2.42   2025: -0.08   2026: +0.21
Five different split points (2022-01 through 2024-01) all show the same
pattern: train negative whenever it includes 2021-2022, test strongly
positive whenever it includes 2023-2024. This ISN'T "the portfolio
generalizes out-of-sample" -- it's that the entire long-run Sharpe is
carried by ONE 2-year window (2023-2024), with two bad years before it
and two flat/mediocre years (2025: -0.08, 2026: +0.21) after it. The most
recent data -- what actually matters for "would this work starting
tomorrow" -- is unimpressive, not strong. Revise the earlier "OOS Sharpe
1.634 looks great" framing accordingly: it was real for 2023-2024
specifically, not a demonstrated durable edge going forward.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backtest.portfolio_diversification_check import (
    cvar,
    get_daily_returns,
    max_drawdown_pct,
    probabilistic_sharpe_ratio,
    risk_summary,
    sharpe_stats,
    sortino_ratio,
)
from backtest.run_vrp import vrp_daily_pnl
from strategies.registry import load_strategies

TRAIN_END = "2023-01-01"  # same split as backtest/tune_composite.py
MAX_LEG_WEIGHT = 0.35  # cap so no single leg (carry) can dominate risk-parity
VOL_LOOKBACK = 30

CANDIDATES = [
    ("niche_dual_thrust", "ohlcv"),
    ("academic_turn_of_month_seasonality", "ohlcv"),
    ("onchain_hash_ribbon", "onchain"),
    ("carry_static_positive_funding", "funding_rate"),
    ("arbitrage_coinbase_premium_trend", "cross_exchange"),
    ("hedge_trend_vol_bear", "ohlcv"),
]
# vol_risk_premium isn't in CANDIDATES/get_daily_returns's dispatch -- its
# data contract (DVOL + realized vol, no strategies/registry.py signals(df))
# doesn't fit the (name, data_type) tuple pattern the others share. Added
# directly below instead, same way diversify_carry.py reuses
# carry_returns_from_funding rather than forcing it through the registry.


def capped_risk_parity_weights(returns_df: pd.DataFrame, vol_lookback: int, max_weight: float) -> pd.DataFrame:
    rolling_vol = returns_df.rolling(vol_lookback).std()
    inv_vol = 1 / rolling_vol.replace(0, np.nan)
    raw_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

    # Clip-and-renormalize: cap the loudest leg, redistribute the excess
    # proportionally to everyone else, repeat until nothing exceeds the cap
    # (a couple of passes always suffices for a handful of legs).
    weights = raw_weights.clip(upper=max_weight)
    for _ in range(10):
        deficit = 1 - weights.sum(axis=1)
        if (deficit.abs() < 1e-9).all():
            break
        room = (max_weight - weights).clip(lower=0)
        room_total = room.sum(axis=1).replace(0, np.nan)
        add = room.div(room_total, axis=0).mul(deficit, axis=0).fillna(0.0)
        weights = (weights + add).clip(upper=max_weight)
    return weights.shift(1)


def report_split(name: str, r: pd.Series) -> None:
    sharpe, skew, kurt, n = sharpe_stats(r)
    psr = probabilistic_sharpe_ratio(sharpe, 0.0, n, skew, kurt) if n > 2 else float("nan")
    print(f"  {name}: sharpe={sharpe:.3f}  n={n}  PSR(vs 0)={psr:.4f}  max_dd={max_drawdown_pct(r):.2f}%")


def main() -> None:
    strategies = {s.meta["name"]: s for s in load_strategies()}

    returns = {name: get_daily_returns(name, dtype, strategies) for name, dtype in CANDIDATES}
    returns["vol_risk_premium"] = vrp_daily_pnl()
    n_legs = len(CANDIDATES) + 1
    returns_df = pd.DataFrame(returns).dropna()
    print(f"Overlapping days across all {n_legs} legs: {len(returns_df)} "
          f"({returns_df.index.min().date()} - {returns_df.index.max().date()})\n")

    print("=== Pairwise correlation of daily returns ===")
    pd.set_option("display.width", 200)
    print(returns_df.corr().round(3).to_string())

    equal_weight = returns_df.mean(axis=1)
    uncapped_rp_weights = (1 / returns_df.rolling(VOL_LOOKBACK).std().replace(0, np.nan))
    uncapped_rp_weights = uncapped_rp_weights.div(uncapped_rp_weights.sum(axis=1), axis=0).shift(1)
    uncapped_rp = (returns_df * uncapped_rp_weights).sum(axis=1, min_count=1).dropna()

    capped_weights = capped_risk_parity_weights(returns_df, VOL_LOOKBACK, MAX_LEG_WEIGHT)
    capped_rp = (returns_df * capped_weights).sum(axis=1, min_count=1).dropna()

    print(f"\nAverage weight per leg — uncapped risk-parity:")
    print(uncapped_rp_weights.mean().round(3).to_string())
    print(f"\nAverage weight per leg — capped risk-parity (max {MAX_LEG_WEIGHT:.0%}/leg):")
    print(capped_weights.mean().round(3).to_string())

    print(f"\n=== Full-period Sharpe: equal-weight vs uncapped RP vs capped RP ({n_legs} legs incl. hedge+VRP) ===")
    report_split("equal_weight", equal_weight)
    report_split("risk_parity_uncapped", uncapped_rp)
    report_split(f"risk_parity_capped_{MAX_LEG_WEIGHT:.0%}", capped_rp)

    # --- Walk-forward: does the capped-RP portfolio's edge survive
    # out-of-sample, or is it another case of composite_tuning_top10_oos.csv
    # (train Sharpe ~1.0 that collapses to 0.07-0.29 out-of-sample)? ---
    train = capped_rp[capped_rp.index < TRAIN_END]
    test = capped_rp[capped_rp.index >= TRAIN_END]
    print(f"\n=== Walk-forward split at {TRAIN_END} — capped risk-parity portfolio ===")
    report_split("train", train)
    report_split("test (out-of-sample)", test)

    train_eq = equal_weight[equal_weight.index < TRAIN_END]
    test_eq = equal_weight[equal_weight.index >= TRAIN_END]
    print(f"\n=== Walk-forward split at {TRAIN_END} — equal-weight portfolio (for comparison) ===")
    report_split("train", train_eq)
    report_split("test (out-of-sample)", test_eq)

    out = returns_df.copy()
    out["portfolio_equal_weight"] = equal_weight
    out["portfolio_risk_parity_uncapped"] = uncapped_rp
    out[f"portfolio_risk_parity_capped_{MAX_LEG_WEIGHT:.0%}"] = capped_rp
    out.to_csv("results/hedged_portfolio_returns.csv")
    print("\nSaved daily returns to results/hedged_portfolio_returns.csv")

    summary_rows = [risk_summary(name, returns_df[name]) for name in returns_df.columns]
    summary_rows.append(risk_summary("portfolio_equal_weight", equal_weight))
    summary_rows.append(risk_summary("portfolio_risk_parity_uncapped", uncapped_rp))
    summary_rows.append(risk_summary(f"portfolio_risk_parity_capped_{MAX_LEG_WEIGHT:.0%}", capped_rp))
    summary = pd.DataFrame(summary_rows)
    print("\n=== Risk summary: Sharpe vs Sortino vs CVaR(95%) vs max drawdown ===")
    print(summary.to_string(index=False))
    summary.to_csv("results/hedged_portfolio_summary.csv", index=False)
    print("\nSaved to results/hedged_portfolio_summary.csv")


if __name__ == "__main__":
    main()
