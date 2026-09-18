"""Run every ohlcv-based strategy across several distinct BTC market
regimes and rank by robustness — a strategy that only performs well in one
regime is a weaker finding than one that holds up across bear/sideways/bull.

Fetches one long OHLCV history (cached) and slices it per regime, rather
than re-fetching per window.

Usage:
    python backtest/run_regimes.py --symbol BTC/USDT --timeframe 1h
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from backtest.run import run_backtest
from data.fetch import fetch_ohlcv

# ponytail: hand-picked regime windows for BTC specifically, not a general
# regime detector. Good enough for "does this hold up across bear/sideways/
# bull", add a real regime classifier (e.g. off realized vol + trend slope)
# if this needs to generalize to other symbols.
REGIMES = {
    "bull_2017": ("2017-08-01", "2017-12-18"),
    "bear_2018": ("2018-01-01", "2018-12-15"),
    "sideways_2019": ("2019-01-01", "2020-01-01"),
    "covid_crash_2020": ("2020-02-01", "2020-04-01"),
    "bull_2020_2021": ("2020-10-01", "2021-04-14"),
    "correction_2021": ("2021-04-14", "2021-07-20"),
    "bull_2021_q4": ("2021-07-20", "2021-11-10"),
    "bear_2022": ("2022-01-01", "2023-01-01"),
    "sideways_2023": ("2023-01-01", "2023-10-15"),
    "bull_2023_2024_etf": ("2023-10-15", "2024-06-01"),
    "bull_2024_2025": ("2024-06-01", None),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--out", default="results/regime_comparison.csv")
    args = parser.parse_args()

    earliest_since = min(start for start, _ in REGIMES.values())
    full_df = fetch_ohlcv(args.symbol, args.timeframe, earliest_since)

    all_results = []
    for regime_name, (start, end) in REGIMES.items():
        window = full_df.loc[start:end] if end else full_df.loc[start:]
        if len(window) < 50:
            print(f"Skipping {regime_name}: only {len(window)} bars available")
            continue
        results = run_backtest(window, fee=args.fee)
        results.insert(0, "regime", regime_name)
        all_results.append(results)

    combined = pd.concat(all_results, ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    regime_names = [r for r in REGIMES if r in combined["regime"].unique()]
    pivot = combined.pivot_table(index="strategy", columns="regime", values="sharpe")
    pivot["avg_sharpe"] = pivot[regime_names].mean(axis=1, skipna=True)
    pivot["regimes_positive"] = (pivot[regime_names] > 0).sum(axis=1)
    robustness = pivot.sort_values(["regimes_positive", "avg_sharpe"], ascending=False)

    robustness_path = out_path.with_name(out_path.stem + "_robustness.csv")
    robustness.to_csv(robustness_path)

    print(f"\n=== Robustness across {len(regime_names)} regimes ({', '.join(regime_names)}) ===")
    print(robustness.to_string())
    print(f"\nSaved to {out_path} and {robustness_path}")


if __name__ == "__main__":
    main()
