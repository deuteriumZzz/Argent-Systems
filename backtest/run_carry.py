"""Run every registered funding-rate carry strategy and rank them.

Different PnL model from backtest/run.py: carry strategies don't earn from
price moves (the position is delta-neutral by construction — long spot +
short perp, or the mirror), they earn the funding payment itself. Position
decided from data up to and including bar t is applied to the funding print
that lands at t+1 (shift(1)) — you can only collect a print you were
already positioned for, funding isn't retroactive.

Usage:
    python backtest/run_carry.py --symbol BTC/USDT:USDT --since 2021-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data.funding import fetch_funding_rate
from strategies.registry import load_strategies

# ponytail: flat per-flip cost approximating two-leg (spot+perp) taker fees
# plus basis slippage on entry/exit; a real desk would model maker fills and
# the spot-vs-perp basis spread separately, add that if this needs to be
# precise rather than comparative.
TAKER_FEE_ROUND_TRIP = 0.0008


def run_carry_backtest(df: pd.DataFrame, fee_per_flip: float = TAKER_FEE_ROUND_TRIP) -> pd.DataFrame:
    rows = []
    for strat in load_strategies():
        if strat.meta.get("data_type") != "funding_rate":
            continue

        weight = strat.signals(df, **strat.meta.get("default_params", {}))
        position = weight.shift(1).fillna(0.0)

        funding_pnl = position * df["funding_rate"]
        flip_cost = position.diff().fillna(position).abs() * fee_per_flip
        period_return = funding_pnl - flip_cost

        equity = (1 + period_return).cumprod()
        total_return_pct = (equity.iloc[-1] - 1) * 100
        drawdown = (equity - equity.cummax()) / equity.cummax()
        max_drawdown_pct = drawdown.min() * 100

        periods_per_year = pd.Timedelta("365d") / (df.index[1] - df.index[0])
        mean, std = period_return.mean(), period_return.std()
        sharpe = (mean / std) * np.sqrt(periods_per_year) if std else float("nan")

        num_position_changes = int((position.diff().fillna(position) != 0).sum())
        rows.append(
            {
                "strategy": strat.meta["name"],
                "category": strat.meta["category"],
                "total_return_pct": total_return_pct,
                "sharpe": sharpe,
                "max_drawdown_pct": max_drawdown_pct,
                "avg_funding_rate_pct": df["funding_rate"].mean() * 100,
                "num_position_changes": num_position_changes,
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binanceusdm")
    parser.add_argument("--since", default="2021-01-01")
    parser.add_argument("--fee-per-flip", type=float, default=TAKER_FEE_ROUND_TRIP)
    parser.add_argument("--out", default="results/carry_comparison.csv")
    args = parser.parse_args()

    df = fetch_funding_rate(args.symbol, args.since, exchange_id=args.exchange)
    results = run_carry_backtest(df, fee_per_flip=args.fee_per_flip)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
