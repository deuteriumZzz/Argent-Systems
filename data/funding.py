"""Perpetual futures funding-rate history, cached to parquet.

Used by strategies/carry_*.py, which trade the delta-neutral basis (long
spot + short perp when funding is positive, or the mirror when it's
negative) to collect the funding payment itself rather than a directional
price move. Separate from data/fetch.py's spot OHLCV because the data
shape and the PnL model built on top of it are both different (see
backtest/run_carry.py).
"""
from __future__ import annotations

from pathlib import Path

import ccxt
import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MAX_REQUESTS = 200


def fetch_funding_rate(
    symbol: str = "BTC/USDT:USDT",
    since: str = "2021-01-01",
    exchange_id: str = "binanceusdm",
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch funding-rate prints for a perpetual `symbol` from `since` to now.

    Returns a DataFrame indexed by UTC timestamp with a single
    `funding_rate` column (the rate paid per funding interval, e.g. 0.0001
    = 0.01% every 8h on Binance). Cached to data/cache/*.parquet.
    """
    safe_symbol = symbol.replace("/", "").replace(":", "_")
    cache_path = CACHE_DIR / f"funding_{exchange_id}_{safe_symbol}_{since}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")

    all_rates: list[dict] = []
    for _ in range(MAX_REQUESTS):
        batch = exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=limit)
        if not batch:
            break
        all_rates.extend(batch)
        since_ms = batch[-1]["timestamp"] + 1
        if len(batch) < limit:
            break

    if not all_rates:
        raise ValueError(f"No funding rate history for {symbol} since {since}")

    df = pd.DataFrame(
        {
            "timestamp": [r["timestamp"] for r in all_rates],
            "funding_rate": [r["fundingRate"] for r in all_rates],
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
    df.to_parquet(cache_path)
    return df
