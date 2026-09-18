"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

Every strategy's raw Sharpe ratio is inflated by two things this script
corrects for:
  1. Selection bias: we tested N strategies, so even with zero real skill
     the best of N noisy backtests will show a positive Sharpe just by
     chance. The "benchmark" isn't 0, it's the expected max Sharpe you'd
     get from N trials of pure luck.
  2. Non-normal returns: the classic Sharpe-ratio significance test
     assumes normally distributed returns. Crypto returns are fat-tailed
     and skewed, so the raw t-stat overstates confidence.

DSR(strategy) = P(true Sharpe > benchmark), using the Probabilistic Sharpe
Ratio formula with the strategy's own skew/kurtosis and sample size, where
benchmark = E[max Sharpe over N trials under the null of no skill].

Usage:
    python backtest/deflated_sharpe.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy.stats import norm

from data.fetch import fetch_ohlcv
from strategies.registry import load_strategies

EULER_MASCHERONI = 0.5772156649015329


def strategy_stats(df: pd.DataFrame, strat, fee: float = 0.001) -> dict | None:
    weight = strat.signals(df, **strat.meta.get("default_params", {}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = vbt.Portfolio.from_orders(
            close=df["close"], size=weight, size_type="targetpercent", fees=fee,
            freq=pd.infer_freq(df.index) or "1d",
        )
    returns = pf.returns()
    if pf.trades.count() == 0 or returns.std() == 0:
        return None
    return {
        "name": strat.meta["name"],
        "category": strat.meta["category"],
        "sharpe": pf.sharpe_ratio(),
        "n": len(returns),
        # pandas' .skew()/.kurtosis() are the sample estimators the DSR
        # paper assumes; .kurtosis() is EXCESS kurtosis (0 for normal), but
        # the formula wants raw kurtosis (3 for normal), hence the +3.
        "skew": returns.skew(),
        "kurt": returns.kurtosis() + 3,
        "trades": pf.trades.count(),
        "total_return_pct": pf.total_return() * 100,
    }


def expected_max_sharpe(sharpe_values: np.ndarray, n_trials: int) -> float:
    """E[max Sharpe over n_trials independent strategies with zero true skill]."""
    var_sr = np.var(sharpe_values, ddof=1)
    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def probabilistic_sharpe_ratio(sharpe_hat: float, sharpe_star: float, n: int, skew: float, kurt: float) -> float:
    numerator = (sharpe_hat - sharpe_star) * np.sqrt(n - 1)
    denominator = np.sqrt(max(1 - skew * sharpe_hat + ((kurt - 1) / 4) * sharpe_hat**2, 1e-12))
    return float(norm.cdf(numerator / denominator))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--since", default="2017-08-01")
    parser.add_argument("--out", default="results/deflated_sharpe.csv")
    args = parser.parse_args()

    df = fetch_ohlcv(args.symbol, args.timeframe, args.since)

    rows = []
    for strat in load_strategies():
        if strat.meta.get("data_type", "ohlcv") != "ohlcv":
            continue
        stats = strategy_stats(df, strat)
        if stats:
            rows.append(stats)

    results = pd.DataFrame(rows)
    n_trials = len(results)
    sr_benchmark = expected_max_sharpe(results["sharpe"].to_numpy(), n_trials)

    results["dsr"] = results.apply(
        lambda r: probabilistic_sharpe_ratio(r["sharpe"], sr_benchmark, r["n"], r["skew"], r["kurt"]),
        axis=1,
    )
    results = results.sort_values("dsr", ascending=False).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    survivors = results[results["dsr"] >= 0.95]
    print(
        f"Trials (N): {n_trials}  |  benchmark = expected max Sharpe under pure luck: {sr_benchmark:.3f} "
        f"(raw Sharpe below this is indistinguishable from noise)\n"
    )
    pd.set_option("display.width", 200)
    print(results.head(20).to_string(index=False))
    print(f"\n{len(survivors)} of {n_trials} strategies survive DSR >= 0.95 after correcting for {n_trials} trials.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
