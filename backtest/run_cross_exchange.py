"""Run every registered cross-exchange-premium strategy on BTC price.

Same price-PnL engine as backtest/run.py (long/short BTC, earn from
`close` moves) — the cross-exchange premium is used as a signal, not as a
literal spread to arbitrage (see data/cross_exchange.py for why). Reuses
run_backtest() filtered to META["data_type"] == "cross_exchange".

Usage:
    python backtest/run_cross_exchange.py --since 2020-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.run import run_backtest
from data.cross_exchange import fetch_cross_exchange_premium


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2020-01-01")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--out", default="results/cross_exchange_comparison.csv")
    args = parser.parse_args()

    df = fetch_cross_exchange_premium(args.since)
    results = run_backtest(df, fee=args.fee, data_type="cross_exchange")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
