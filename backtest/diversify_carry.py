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
history, AND fixing this script's own dropna()-intersection bug, which
first reported a too-good Sharpe 9.25/max_dd -0.61% by silently shrinking
the whole panel to Bybit's shorter, calmer 471-day window): with each leg
weighted over its own real full history (union of dates, skipna mean, not
intersection), the honest result is the SAME shape as the pre-Bybit-fix
3-leg finding — DIVERSIFIED sharpe=2.877, max_dd=-12.93% over the full
2087 days. Real 6-leg correlation: BTC 0.76 cross-exchange, 0.51-0.72
cross-asset — genuine moderate diversification on both axes. But
binanceusdm:SOL's own real max_dd is -35.36% over its own full history
(a real tail event BTC/ETH never had), which is why the diversified
sleeve's baseline drawdown doesn't actually improve much versus BTC alone
(-0.41%) despite lower correlation — diversifying trades one exchange's
tail risk for importing SOL's own real historical one, not a free
reduction.

WHO WAS ON THE OTHER SIDE OF SOL'S DRAWDOWN, specifically (not a generic
"funding can go negative" story): the underwater period runs 2022-04-30 to
2023-01-13, with its sharpest leg during the FTX collapse week
(2022-11-08 to 11-14) — SOL funding averaged -0.46%/8h that week and
printed -2.00%/8h (Binance's exchange-enforced floor for SOL, pinned
repeatedly) vs BTC's worst print the same week of only -0.12%/8h, 16.7x
milder. SOL was FTX/Alameda's signature token; the panic was concentrated
in it specifically, not the market broadly. Two implications: (1) an
asset whose fortunes are unusually tied to a single counterparty can see
funding pinned at the exchange's actual floor for days, not just dip
below whatever a synthetic stress test assumed — stress_test_carry.py's
-0.8%/8h default is 2.5x MILDER than what SOL genuinely printed here, so
that script's "6.7x worse than history" framing undersold the real tail;
(2) this is exactly the single-counterparty concentration risk carry
diversification is supposed to reduce, so it isn't a reason to avoid
diversifying into SOL — it's the reason the whole sleeve still needs a
hedge/cap, same as everything else in this project. The injected-shock comparison now uses real overlapping data
(2024-06 is well inside every leg's history): single BTC leg -28.63%
during the shock vs the diversified sleeve's +0.49% (no shock) / -10.25%
(shocked) — diversification cuts this specific shock's damage
proportionally, same conclusion as before the union-of-dates fix, just
now on a bug-free number instead of a lucky one.
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

    # Union of dates (outer join), NOT intersection: a leg that starts later
    # (Bybit, 2021-09) shouldn't shrink the whole panel down to its own
    # shorter history. Correlation and Sharpe below use each leg's own full
    # available range (pandas .corr()/.mean() already skip NaN pairwise).
    returns_df = pd.DataFrame(returns).sort_index()
    intersection = returns_df.dropna()
    print(f"\nFull date range across {len(returns_df.columns)} legs: {len(returns_df)} days "
          f"({returns_df.index.min().date()} - {returns_df.index.max().date()}); "
          f"{len(intersection)} days where all {len(returns_df.columns)} legs have data\n")

    print("=== Pairwise correlation (pairwise-complete, not restricted to the full intersection) ===")
    pd.set_option("display.width", 200)
    print(returns_df.corr().round(3).to_string())

    # Equal-weight across whichever legs actually have data that day, so a
    # leg's absence before it existed doesn't get silently treated as a 0%
    # return (which would understate, not just dilute, the sleeve).
    equal_weight = returns_df.mean(axis=1, skipna=True)

    print(f"\n=== Individual legs (own full history) vs diversified equal-weight sleeve ===")
    for col in returns_df.columns:
        s, _, _, n = sharpe_stats(returns_df[col].dropna())
        print(f"  {col}: sharpe={s:.3f}  n={n}  max_dd={max_drawdown_pct(returns_df[col].dropna()):.2f}%")
    ew_sharpe, _, _, ew_n = sharpe_stats(equal_weight)
    print(f"  DIVERSIFIED (equal-weight, skipna): "
          f"sharpe={ew_sharpe:.3f}  n={ew_n}  max_dd={max_drawdown_pct(equal_weight):.2f}%")

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
        stressed_df[baseline_btc_col] = stressed_btc.reindex(stressed_df.index)
        stressed_equal_weight = stressed_df.mean(axis=1, skipna=True)

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

    summary_rows = [risk_summary(name, returns_df[name].dropna()) for name in returns_df.columns]
    summary_rows.append(risk_summary("diversified_equal_weight", equal_weight))
    pd.DataFrame(summary_rows).to_csv("results/diversified_carry_summary.csv", index=False)
    print("\nSaved to results/diversified_carry_returns.csv and results/diversified_carry_summary.csv")


if __name__ == "__main__":
    main()
