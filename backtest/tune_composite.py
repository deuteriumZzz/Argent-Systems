"""Walk-forward parameter search for custom_regime_adaptive_composite.

Splits BTC history at TRAIN_END: grid-searches parameters on the train
window only, then reports how the best-in-sample combos actually perform
on the untouched test window. If in-sample rank doesn't predict
out-of-sample performance, that's the overfitting warning working as
intended — not a bug in the search.

Usage:
    python backtest/tune_composite.py
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import vectorbt as vbt

from data.fetch import fetch_ohlcv
from data.onchain import fetch_onchain_metrics
from strategies.custom_regime_adaptive_composite import signals

TRAIN_END = "2023-01-01"  # train: bull_2017 .. bear_2022 (7 regimes). test: sideways_2023 onward (3 regimes), never seen by the search.

PARAM_GRID = {
    "vol_spike_threshold": [1.0, 1.5, 2.0],
    "hash_fast": [20, 30],
    "hash_slow": [60, 90],
    "k1": [0.3, 0.5, 0.7],
    "k2": [0.3, 0.5, 0.7],
    "recovery_window": [7, 14],
}


def sharpe_for(df: pd.DataFrame, params: dict, fee: float = 0.001) -> tuple[float, float, int]:
    weight = signals(df, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = vbt.Portfolio.from_orders(
            close=df["close"], size=weight, size_type="targetpercent", fees=fee, freq="1d"
        )
    n = pf.trades.count()
    sharpe = pf.sharpe_ratio() if n else float("nan")
    return sharpe, pf.total_return() * 100, n


def main() -> None:
    price_df = fetch_ohlcv("BTC/USDT", "1d", "2017-08-01")
    onchain_df = fetch_onchain_metrics("2017-08-01")
    df = price_df.join(onchain_df.reindex(price_df.index, method="ffill"))
    df = df.dropna(subset=list(onchain_df.columns))

    train_df = df.loc[:TRAIN_END]
    test_df = df.loc[TRAIN_END:]

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    print(f"Searching {len(combos)} parameter combinations on train data ({train_df.index.min().date()} - {train_df.index.max().date()})...")

    rows = []
    for values in combos:
        params = dict(zip(keys, values))
        train_sharpe, train_return, train_trades = sharpe_for(train_df, params)
        rows.append({**params, "train_sharpe": train_sharpe, "train_return_pct": train_return, "train_trades": train_trades})

    results = pd.DataFrame(rows).dropna(subset=["train_sharpe"]).sort_values("train_sharpe", ascending=False)

    top10 = results.head(10).copy()
    test_sharpes, test_returns, test_trades = [], [], []
    # to_dict("records") keeps each column's own dtype; iterrows() would
    # upcast the int params (hash_fast etc.) to float64 since a Series row
    # can only hold one dtype, and rolling() rejects a float window size.
    for row in top10.to_dict("records"):
        params = {k: row[k] for k in keys}
        s, r, n = sharpe_for(test_df, params)
        test_sharpes.append(s)
        test_returns.append(r)
        test_trades.append(n)
    top10["test_sharpe"] = test_sharpes
    top10["test_return_pct"] = test_returns
    top10["test_trades"] = test_trades

    out_path = Path("results/composite_tuning.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    top10.to_csv("results/composite_tuning_top10_oos.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n=== Top 10 by train Sharpe, with out-of-sample ({test_df.index.min().date()} - {test_df.index.max().date()}) check ===")
    print(top10.to_string(index=False))

    default_params = {
        "vol_spike_threshold": 1.5, "hash_fast": 30, "hash_slow": 60,
        "k1": 0.5, "k2": 0.5, "recovery_window": 14,
    }
    default_train = sharpe_for(train_df, default_params)
    default_test = sharpe_for(test_df, default_params)
    print(f"\nFor comparison, the untuned defaults: train_sharpe={default_train[0]:.3f}, test_sharpe={default_test[0]:.3f}")
    print(f"\nSaved to {out_path} and results/composite_tuning_top10_oos.csv")


if __name__ == "__main__":
    main()
