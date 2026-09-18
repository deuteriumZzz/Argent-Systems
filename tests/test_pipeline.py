"""Self-check: every registered strategy runs end-to-end on synthetic data.

No network calls (doesn't touch data/fetch.py's Binance call) so it runs
anywhere. Run with: python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backtest.run import run_backtest
from backtest.run_carry import run_carry_backtest
from strategies.registry import load_strategies

ONCHAIN_COLUMNS = [
    "unique_addresses",
    "hash_rate",
    "miners_revenue_usd",
    "transaction_volume_usd",
    "market_cap_usd",
]


def make_synthetic_ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.01, n)
    close = 20000 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.uniform(10, 100, n)
    index = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def make_synthetic_funding(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    funding_rate = rng.normal(0.0001, 0.0003, n)
    index = pd.date_range("2022-01-01", periods=n, freq="8h", tz="UTC")
    return pd.DataFrame({"funding_rate": funding_rate}, index=index)


def make_synthetic_onchain(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """OHLCV columns plus on-chain columns, all on a daily index (matches
    how backtest/run_onchain.py merges blockchain.info data onto price)."""
    rng = np.random.default_rng(seed)
    df = make_synthetic_ohlcv(n=n, seed=seed)
    df.index = pd.date_range("2022-01-01", periods=n, freq="1d", tz="UTC")
    df["unique_addresses"] = rng.uniform(3e5, 6e5, n)
    df["hash_rate"] = rng.uniform(1e8, 3e8, n)
    df["miners_revenue_usd"] = rng.uniform(2e7, 5e7, n)
    df["transaction_volume_usd"] = rng.uniform(1e9, 5e9, n)
    df["market_cap_usd"] = df["close"] * 19_000_000
    return df


def make_synthetic_cross_exchange(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """BTC close plus a synthetic cross-exchange premium column, daily
    (matches backtest/run_cross_exchange.py's data shape)."""
    rng = np.random.default_rng(seed)
    df = make_synthetic_ohlcv(n=n, seed=seed)[["close"]].copy()
    df.index = pd.date_range("2022-01-01", periods=n, freq="1d", tz="UTC")
    df["coinbase_premium_pct"] = rng.normal(0, 0.15, n)
    return df


def main() -> None:
    strategies = load_strategies()
    assert len(strategies) >= 1, "no strategies discovered"

    data_by_type = {
        "ohlcv": make_synthetic_ohlcv(),
        "funding_rate": make_synthetic_funding(),
        "onchain": make_synthetic_onchain(),
        "cross_exchange": make_synthetic_cross_exchange(),
    }

    for strat in strategies:
        for key in ("name", "category", "source", "license", "description", "default_params"):
            assert key in strat.meta, f"{strat.meta.get('name', '?')} missing META['{key}']"

        data_type = strat.meta.get("data_type", "ohlcv")
        df = data_by_type[data_type]
        weight = strat.signals(df, **strat.meta["default_params"])
        assert isinstance(weight, pd.Series), f"{strat.meta['name']}.signals must return a Series"
        assert weight.index.equals(df.index), f"{strat.meta['name']}.signals index must match input"
        assert weight.notna().all(), f"{strat.meta['name']}.signals produced NaN weights"
        assert weight.between(-1.0, 1.0).all(), f"{strat.meta['name']}.signals weight out of [-1, 1]"

    counts = {
        dt: sum(1 for s in strategies if s.meta.get("data_type", "ohlcv") == dt) for dt in data_by_type
    }

    results_by_type = {
        "ohlcv": run_backtest(data_by_type["ohlcv"]),
        "funding_rate": run_carry_backtest(data_by_type["funding_rate"]),
        "onchain": run_backtest(data_by_type["onchain"], data_type="onchain"),
        "cross_exchange": run_backtest(data_by_type["cross_exchange"], data_type="cross_exchange"),
    }
    for dt, results in results_by_type.items():
        assert len(results) == counts[dt], f"{dt}: expected {counts[dt]} results, got {len(results)}"

    summary = ", ".join(f"{counts[dt]} {dt}" for dt in data_by_type)
    print(f"OK: {len(strategies)} strategies loaded and backtested on synthetic data ({summary}).")
    for results in results_by_type.values():
        if len(results):
            print(results.to_string(index=False))


if __name__ == "__main__":
    main()
