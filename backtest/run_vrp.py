"""Volatility risk premium (VRP) on BTC: is implied vol (DVOL) systematically
higher than realized vol, and does a simplified short-variance-swap proxy
actually monetize that gap?

Who's on the other side: option buyers who want convex downside/gap
protection pay a persistent premium over what volatility actually
realizes -- a well-documented risk premium in both traditional and crypto
options markets (the buyer is paying for insurance, the seller is
underwriting it). Reimplemented from the standard concept, no code
borrowed.
License: N/A (concept).

DIFFERENT DATA CONTRACT: uses data/deribit_vol.py (DVOL, not OHLCV) joined
against BTC daily closes from data/fetch.py. Not wired into
strategies/registry.py's signals(df) contract like the OHLCV/onchain/
funding/cross_exchange strategies -- variance-swap payoff needs BOTH a
fixed implied-vol leg and a realized-vol leg computed independently, which
doesn't fit a single signals(df) -> weight function. Run directly.

METHOD: rather than modeling actual option Greeks/strikes (which needs
data we don't have -- the full paywalled options chain), this proxies a
short variance swap's daily mark-to-market: each day accrues
(implied_variance - that_day's_annualized_realized_variance) / 365. This
is the standard way quant desks think about VRP capture in isolation from
delta-hedging noise -- see e.g. the CBOE/Cboe VRP literature for the
equity-market equivalent. It is still a simplification: real variance
swaps have basis risk against a replicating options portfolio, and this
carries none of that.

Usage:
    python backtest/run_vrp.py

RESULT (2021-03-24 to 2026-09-18, n=2005 days): the premium is real and
was the strongest risk-adjusted result in this whole project --
sharpe=2.513, PSR(vs 0)=1.0000, cumulative +50.7% of notional over 5.5
years, max_dd only -3.97%. Worst single days line up with real,
identifiable crash events (2022-06-13 Celsius/3AC, 2022-11-09 FTX
collapse, 2021-05-19 May 2021 crash) -- confirms this is capturing genuine
short-vol tail risk, not a modeling artifact.

CAVEAT that matters more than the headline number: the premium has been
compressing every single year (mean daily pnl 7.48bps in 2021 -> 1.21bps
in 2025 -> NEGATIVE -0.49bps in 2026 year-to-date) as the "who's on the
other side" story predicts -- more participants systematically selling
vol competes the premium away. The historical Sharpe is real; whether
it's still capturable going forward, on this evidence, is doubtful.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from backtest.portfolio_diversification_check import (
    max_drawdown_pct,
    probabilistic_sharpe_ratio,
    sharpe_stats,
)
from data.deribit_vol import fetch_dvol
from data.fetch import fetch_ohlcv

START = "2021-01-01"


def main() -> None:
    dvol = fetch_dvol("BTC")
    iv = dvol["close"] / 100.0  # decimal annualized IV

    btc = fetch_ohlcv("BTC/USDT", "1d", START)
    btc.index = pd.to_datetime(btc.index, utc=True).normalize()
    log_ret = np.log(btc["close"]).diff()

    df = pd.DataFrame({"iv": iv, "log_ret": log_ret}).dropna()

    implied_var = df["iv"] ** 2
    realized_var_daily = (df["log_ret"] ** 2) * 365
    daily_pnl = (implied_var - realized_var_daily) / 365

    sharpe, skew, kurt, n = sharpe_stats(daily_pnl)
    psr = probabilistic_sharpe_ratio(sharpe, 0.0, n, skew, kurt)
    print(f"Short-variance-swap proxy, {n} days ({df.index.min().date()} - {df.index.max().date()})")
    print(f"sharpe={sharpe:.3f}  skew={skew:.3f}  kurt={kurt:.3f}  PSR(vs 0)={psr:.4f}")
    print(f"cumulative return: {daily_pnl.sum() * 100:.1f}% of notional")
    print(f"worst single day: {daily_pnl.min() * 100:.2f}%  on {daily_pnl.idxmin().date()}")

    print("\nWorst 10 days (short-vol tail risk):")
    print((daily_pnl * 100).sort_values().head(10).to_string())

    print("\nBy year (decay check):")
    grp = daily_pnl.groupby(daily_pnl.index.year)
    yearly = pd.DataFrame({
        "mean_bps": grp.mean() * 10_000,
        "worst_day_bps": grp.min() * 10_000,
        "sum_pct": grp.sum() * 100,
    })
    print(yearly.to_string())

    out = pd.DataFrame({"iv": df["iv"], "log_ret": df["log_ret"], "daily_pnl_pct_notional": daily_pnl})
    out.to_csv("results/vrp_short_variance_returns.csv")
    print("\nSaved to results/vrp_short_variance_returns.csv")


if __name__ == "__main__":
    main()
