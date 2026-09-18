"""Run every registered strategy over the same OHLCV data and rank them.

Usage:
    python backtest/run.py --symbol BTC/USDT --timeframe 1h --since 2021-01-01
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# Allow running as a script (python backtest/run.py) without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import vectorbt as vbt

from data.fetch import fetch_ohlcv
from strategies.registry import load_strategies


def run_backtest(df: pd.DataFrame, fee: float = 0.001, data_type: str = "ohlcv") -> pd.DataFrame:
    """Backtest every strategy whose META["data_type"] matches `data_type`.

    Reused as-is by backtest/run_onchain.py: onchain_*.py strategies still
    size a directional BTC position and earn from `df["close"]` moves, just
    like ohlcv_* ones — only the *input* columns differ (on-chain metrics
    instead of/alongside plain OHLCV), so the same price-PnL engine applies.
    carry_*.py is the one genuinely different PnL model (funding payments,
    not price moves) — see backtest/run_carry.py for that one.
    """
    rows = []
    for strat in load_strategies():
        if strat.meta.get("data_type", "ohlcv") != data_type:
            continue
        weight = strat.signals(df, **strat.meta.get("default_params", {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pf = vbt.Portfolio.from_orders(
                close=df["close"],
                size=weight,
                size_type="targetpercent",
                fees=fee,
                freq=pd.infer_freq(df.index) or "1h",
            )
        num_trades = pf.trades.count()
        # A strategy that never traded has zero return variance, which makes
        # vectorbt's Sharpe formula divide by zero -> +-inf. That is not a
        # real "infinitely good/bad" score, so treat it as undefined instead
        # of letting it sort to the top of the comparison table.
        sharpe = pf.sharpe_ratio() if num_trades else float("nan")
        rows.append(
            {
                "strategy": strat.meta["name"],
                "category": strat.meta["category"],
                "total_return_pct": pf.total_return() * 100,
                "sharpe": sharpe,
                "max_drawdown_pct": pf.max_drawdown() * 100,
                "win_rate_pct": pf.trades.win_rate() * 100 if num_trades else float("nan"),
                "num_trades": num_trades,
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--since", default="2021-01-01")
    parser.add_argument("--fee", type=float, default=0.001, help="round-trip fee fraction per order")
    parser.add_argument("--out", default="results/comparison.csv")
    args = parser.parse_args()

    df = fetch_ohlcv(args.symbol, args.timeframe, args.since)
    results = run_backtest(df, fee=args.fee)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
