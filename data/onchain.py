"""Free daily on-chain metrics for Bitcoin, from blockchain.info's public
Charts API (no auth/API key required).

Bitcoin-only — blockchain.info doesn't track other chains, so strategies
built on this (strategies/onchain_*.py, META["data_type"] = "onchain") only
make sense on a BTC symbol. See backtest/run_onchain.py, which merges this
with regular OHLCV so onchain strategies can still size a directional BTC
position and get backtested through the normal price-PnL engine.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CHARTS = {
    "unique_addresses": "n-unique-addresses",
    "hash_rate": "hash-rate",
    "miners_revenue_usd": "miners-revenue",
    "transaction_volume_usd": "estimated-transaction-volume-usd",
    "market_cap_usd": "market-cap",
}


def fetch_onchain_metrics(since: str = "2018-01-01") -> pd.DataFrame:
    """Fetch daily BTC on-chain metrics from `since` to now.

    Returns a DataFrame indexed by UTC date (midnight) with one column per
    entry in CHARTS. Cached to data/cache/*.parquet.
    """
    cache_path = CACHE_DIR / f"onchain_btc_{since}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    columns = {}
    for column, chart in CHARTS.items():
        resp = requests.get(
            f"https://api.blockchain.info/charts/{chart}",
            params={"start": since, "format": "json", "timespan": "all"},
            timeout=30,
        )
        resp.raise_for_status()
        points = resp.json()["values"]
        columns[column] = pd.Series(
            {pd.to_datetime(p["x"], unit="s", utc=True).normalize(): p["y"] for p in points}
        )

    df = pd.DataFrame(columns).sort_index()
    df = df[df.index >= pd.Timestamp(since, tz="UTC")]
    # Different blockchain.info charts don't all print on every calendar
    # day (e.g. hash-rate updates on its own cadence), which left ~50% NaN
    # gaps when combined naively. These are slow-moving network
    # fundamentals, so carrying the last known value forward across a
    # missing day or two is a reasonable simplification, not fabrication.
    df = df.asfreq("D").ffill()
    df.to_parquet(cache_path)
    return df
