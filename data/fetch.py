"""OHLCV fetcher with local parquet cache, backed by ccxt.

Used by backtest/run.py to pull historical candles for every strategy run
without re-downloading the same range twice.
"""
from __future__ import annotations

from pathlib import Path

import ccxt
import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ponytail: naive full-history refetch on cache miss, no incremental append.
# Fine at pilot scale (single symbol/timeframe); switch to incremental
# top-up once daily re-runs on many symbols make full refetch too slow.
MAX_REQUESTS = 500


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    since: str,
    exchange_id: str = "binance",
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch OHLCV candles for `symbol` from `since` (ISO date) to now.

    Returns a DataFrame indexed by UTC timestamp with columns
    open/high/low/close/volume. Cached to data/cache/*.parquet.
    """
    cache_key = f"{exchange_id}_{symbol.replace('/', '')}_{timeframe}_{since}.parquet"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")

    all_candles: list[list[float]] = []
    for _ in range(MAX_REQUESTS):
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        if not candles:
            break
        all_candles.extend(candles)
        since_ms = candles[-1][0] + 1
        if len(candles) < limit:
            break

    if not all_candles:
        raise ValueError(f"No candles returned for {symbol} {timeframe} since {since}")

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    df.to_parquet(cache_path)
    return df
