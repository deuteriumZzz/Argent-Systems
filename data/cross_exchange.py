"""Cross-exchange price premium (e.g. the well-known "Coinbase Premium").

Fetches BTC OHLCV from two exchanges via ccxt and computes the % premium
of one over the other. This is NOT a literal arbitrage-execution model —
at daily/hourly granularity the backtest can't capture a spread that real
arbitrageurs close within seconds. Instead this treats the premium as a
market-sentiment/flow signal (documented publicly by CryptoQuant/Kaiko as
the "Coinbase Premium Index" — a proxy for US institutional buying
pressure), which is realistic to backtest on daily closes.
"""
from __future__ import annotations

from pathlib import Path

import ccxt
import pandas as pd

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _fetch_daily_close(exchange_id: str, symbol: str, since: str) -> pd.Series:
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")
    all_candles: list[list[float]] = []
    for _ in range(500):
        candles = exchange.fetch_ohlcv(symbol, "1d", since=since_ms, limit=300)
        if not candles:
            break
        all_candles.extend(candles)
        since_ms = candles[-1][0] + 1
        if len(candles) < 300:
            break
    index = pd.to_datetime([c[0] for c in all_candles], unit="ms", utc=True).normalize()
    return pd.Series([c[4] for c in all_candles], index=index).groupby(level=0).last()


def fetch_cross_exchange_premium(
    since: str = "2020-01-01",
    reference_exchange: str = "binance",
    reference_symbol: str = "BTC/USDT",
    premium_exchange: str = "coinbase",
    premium_symbol: str = "BTC/USD",
) -> pd.DataFrame:
    """Daily close from `reference_exchange` plus the % premium of
    `premium_exchange`'s close over it. Cached to data/cache/*.parquet."""
    cache_path = CACHE_DIR / (
        f"cross_exchange_{reference_exchange}_{premium_exchange}_{since}.parquet"
    )
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    reference_close = _fetch_daily_close(reference_exchange, reference_symbol, since)
    premium_close = _fetch_daily_close(premium_exchange, premium_symbol, since)

    df = pd.DataFrame({"close": reference_close}).join(
        premium_close.rename("premium_exchange_close"), how="inner"
    )
    df["coinbase_premium_pct"] = (
        (df["premium_exchange_close"] - df["close"]) / df["close"] * 100
    )
    df.to_parquet(cache_path)
    return df
