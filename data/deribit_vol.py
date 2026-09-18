"""BTC DVOL (Deribit implied volatility index) history, cached to parquet.

Free CSV from CryptoDataDownload (https://www.cryptodatadownload.com/data/deribit/)
-- Deribit's own public API only serves ~2 weeks of realized vol, not a
multi-year implied-vol series, and the full options chain there is paywalled.
This is the one column CryptoDataDownload doesn't paywall.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DVOL_URL = "https://www.cryptodatadownload.com/cdd/DeriBit_volatility_OHLC_BTC.csv"


def fetch_dvol(symbol: str = "BTC") -> pd.DataFrame:
    """Fetch daily DVOL (annualized implied vol, in percent) history.

    Returns a DataFrame indexed by UTC date with open/high/low/close
    columns (close is what strategies should use). Cached to
    data/cache/dvol_{symbol}.parquet.
    """
    cache_path = CACHE_DIR / f"dvol_{symbol}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    resp = requests.get(DVOL_URL, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text), skiprows=1)
    df = df[df["symbol"] == symbol]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()[["open", "high", "low", "close"]]
    df.to_parquet(cache_path)
    return df
