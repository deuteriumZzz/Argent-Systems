"""Run every registered on-chain strategy on BTC price + on-chain metrics.

On-chain metrics from blockchain.info are daily and BTC-only, so this
always fetches daily BTC/USDT OHLCV and forward-fills the (also daily, but
independently-timestamped) on-chain columns onto that index. PnL still
comes from BTC price moves — reuses backtest/run.py's run_backtest(), just
filtered to META["data_type"] == "onchain".

Usage:
    python backtest/run_onchain.py --since 2020-01-01
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.run import run_backtest
from data.fetch import fetch_ohlcv
from data.onchain import fetch_onchain_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2020-01-01")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--out", default="results/onchain_comparison.csv")
    args = parser.parse_args()

    price_df = fetch_ohlcv("BTC/USDT", "1d", args.since)
    onchain_df = fetch_onchain_metrics(args.since)

    df = price_df.join(onchain_df.reindex(price_df.index, method="ffill"))
    df = df.dropna(subset=list(onchain_df.columns))

    results = run_backtest(df, fee=args.fee, data_type="onchain")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
