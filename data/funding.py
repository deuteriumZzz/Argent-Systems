"""Perpetual futures funding-rate history, cached to parquet.

Used by strategies/carry_*.py, which trade the delta-neutral basis (long
spot + short perp when funding is positive, or the mirror when it's
negative) to collect the funding payment itself rather than a directional
price move. Separate from data/fetch.py's spot OHLCV because the data
shape and the PnL model built on top of it are both different (see
backtest/run_carry.py).

FIXED (found via backtest/diversify_carry.py): the pagination loop used to
stop as soon as a batch came back shorter than the requested `limit` —
correct for binanceusdm, wrong for Bybit, whose fetchFundingRateHistory
silently caps each call's page size below whatever `limit` was requested,
so the very first page already looked "short" and the loop quit after 2
months of data instead of ~5 years. Now advances strictly by timestamp
progress instead of batch size.
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
    now_ms = exchange.milliseconds()

    # NOTE: `len(batch) < limit` is NOT a reliable "no more data" signal —
    # some exchanges (Bybit confirmed) silently cap this endpoint's actual
    # per-call page size below whatever `limit` was requested, so a full
    # page from Bybit already looks "short" relative to `limit=1000` and
    # would stop the loop after the very first request. Advance strictly
    # by timestamp progress instead: only stop when a batch is empty, or
    # when it's caught up to the current time, or when it fails to advance
    # (stuck returning the same page).
    all_rates: list[dict] = []
    for _ in range(MAX_REQUESTS):
        batch = exchange.fetch_funding_rate_history(symbol, since=since_ms, limit=limit)
        if not batch:
            break
        all_rates.extend(batch)
        last_ts = batch[-1]["timestamp"]
        if last_ts <= since_ms or last_ts >= now_ms:
            break
        since_ms = last_ts + 1

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
