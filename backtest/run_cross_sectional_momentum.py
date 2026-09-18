"""Cross-sectional (multi-asset) momentum: rank a universe of alts by
trailing return, long the top decile / short the bottom decile, rebalance
weekly.

Who's on the other side: every strategy in this project so far times ONE
asset (BTC) across time — this is the first one to bet across a SECTION of
assets at a fixed point in time instead. The academic case (Liu &
Tsyvinski 2021, "Risks and Returns of Cryptocurrency"; Tzouvanas et al.
2020) is that retail attention diffuses slowly and unevenly across a
fragmented, low-analyst-coverage asset class — the crowd chases whichever
coin is already pumping on social feeds and is slow to rotate out of
laggards, so relative strength persists for weeks rather than getting
arbitraged away instantly. That mechanism is structurally different from
single-asset technical momentum (which this project's own results show
has no edge on BTC alone) — it doesn't require BTC's price history to
contain a repeatable pattern, only that some alts are, for now, more/less
in favor than others.

Reimplemented from the published methodology, no code borrowed.
License: N/A (academic concept, reimplemented from the published rule).

Universe/lookback/rebalance period below are picked from the published
methodology (roughly weekly rebalance, weeks-scale lookback), not fit to
this data — see backtest/tune_composite.py's walk-forward warning for why
that distinction matters.

Usage:
    python backtest/run_cross_sectional_momentum.py

RESULT (2021-01-01 to 2026-09, 30-symbol universe, weekly rebalance): the
long/short spread has no edge (full-period Sharpe 0.04, walk-forward
out-of-sample Sharpe -0.05 — the ranking itself provides nothing). The
long-only top-30% momentum basket looks better in isolation (Sharpe 0.50,
+68% return) but a naive equal-weight basket of ALL 30 assets with NO
ranking at all beats it outright (Sharpe 0.67, +247.5% return) — so the
apparent "momentum edge" was just alt-market beta exposure during a rising
period, not a real cross-sectional effect. The "who's on the other side"
story (slow attention diffusion) didn't survive contact with the data,
same conclusion as most single-asset strategies in this project.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm

from data.fetch import fetch_ohlcv

START = "2021-01-01"
TIMEFRAME = "1d"
LOOKBACK_DAYS = 30
REBALANCE_DAYS = 7
LONG_FRACTION = 0.3  # top 30% of the universe by trailing return
SHORT_FRACTION = 0.3  # bottom 30%
FEE = 0.001  # per rebalance turnover, one-way
TRAIN_END = "2023-01-01"  # same split convention as tune_composite.py

UNIVERSE = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT",
    "DOGE/USDT", "DOT/USDT", "MATIC/USDT", "LTC/USDT", "LINK/USDT", "UNI/USDT",
    "ATOM/USDT", "ETC/USDT", "XLM/USDT", "TRX/USDT", "EOS/USDT", "XMR/USDT",
    "FIL/USDT", "AAVE/USDT", "ALGO/USDT", "VET/USDT", "THETA/USDT", "XTZ/USDT",
    "NEO/USDT", "MKR/USDT", "COMP/USDT", "ZEC/USDT", "DASH/USDT", "ICX/USDT",
]


def fetch_universe() -> dict[str, pd.DataFrame]:
    out = {}
    for sym in UNIVERSE:
        try:
            df = fetch_ohlcv(sym, TIMEFRAME, START)
            if len(df) > LOOKBACK_DAYS + REBALANCE_DAYS:
                out[sym] = df
        except Exception as e:  # noqa: BLE001 - some symbols may not exist/delisted
            print(f"  skip {sym}: {e}")
    return out


def build_close_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {sym: df["close"] for sym, df in data.items()}
    panel = pd.DataFrame(closes).sort_index()
    return panel


def cross_sectional_returns(panel: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Returns (long_short_daily_returns, long_only_daily_returns, avg holdings per rebalance)."""
    rebalance_dates = panel.index[LOOKBACK_DAYS::REBALANCE_DAYS]
    daily_returns = panel.pct_change()

    ls_returns = pd.Series(0.0, index=panel.index)
    lo_returns = pd.Series(0.0, index=panel.index)
    holdings_log = []

    for i, reb_date in enumerate(rebalance_dates):
        window_start = panel.index[panel.index.get_loc(reb_date) - LOOKBACK_DAYS]
        mom = panel.loc[reb_date] / panel.loc[window_start] - 1
        mom = mom.dropna()
        if len(mom) < 10:
            continue

        n_long = max(1, int(len(mom) * LONG_FRACTION))
        n_short = max(1, int(len(mom) * SHORT_FRACTION))
        ranked = mom.sort_values(ascending=False)
        longs = ranked.index[:n_long]
        shorts = ranked.index[-n_short:]
        holdings_log.append({"date": reb_date, "n_assets": len(mom), "longs": list(longs), "shorts": list(shorts)})

        next_reb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else panel.index[-1]
        period_mask = (panel.index > reb_date) & (panel.index <= next_reb)
        period_dates = panel.index[period_mask]
        if len(period_dates) == 0:
            continue

        long_leg = daily_returns.loc[period_dates, longs].mean(axis=1)
        short_leg = daily_returns.loc[period_dates, shorts].mean(axis=1)
        ls_returns.loc[period_dates] = long_leg - short_leg
        lo_returns.loc[period_dates] = long_leg

        # Turnover cost charged once per rebalance, amortized on the first day of the period
        turnover_cost = 2 * FEE  # close old basket, open new basket
        ls_returns.loc[period_dates[0]] -= turnover_cost
        lo_returns.loc[period_dates[0]] -= FEE

    holdings_df = pd.DataFrame(holdings_log)
    return ls_returns, lo_returns, holdings_df


def sharpe_stats(r: pd.Series, periods_per_year: int = 365) -> tuple[float, float, float, int]:
    mean, std = r.mean(), r.std()
    sharpe = (mean / std) * np.sqrt(periods_per_year) if std else float("nan")
    return sharpe, r.skew(), r.kurtosis() + 3, len(r)


def probabilistic_sharpe_ratio(sharpe_hat: float, sharpe_star: float, n: int, skew: float, kurt: float) -> float:
    numerator = (sharpe_hat - sharpe_star) * np.sqrt(n - 1)
    denominator = np.sqrt(max(1 - skew * sharpe_hat + ((kurt - 1) / 4) * sharpe_hat**2, 1e-12))
    return float(norm.cdf(numerator / denominator))


def max_drawdown_pct(r: pd.Series) -> float:
    equity = (1 + r).cumprod()
    return float(((equity - equity.cummax()) / equity.cummax()).min() * 100)


def report(name: str, r: pd.Series) -> None:
    sharpe, skew, kurt, n = sharpe_stats(r)
    psr = probabilistic_sharpe_ratio(sharpe, 0.0, n, skew, kurt) if n > 2 else float("nan")
    print(f"  {name}: sharpe={sharpe:.3f}  n={n}  PSR(vs 0)={psr:.4f}  "
          f"total_return={((1 + r).prod() - 1) * 100:.1f}%  max_dd={max_drawdown_pct(r):.1f}%")


def main() -> None:
    print(f"Fetching universe ({len(UNIVERSE)} candidates)...")
    data = fetch_universe()
    print(f"Usable: {len(data)}/{len(UNIVERSE)} symbols with enough history\n")

    panel = build_close_panel(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ls_returns, lo_returns, holdings = cross_sectional_returns(panel)

    ls_returns = ls_returns.dropna()
    lo_returns = lo_returns.dropna()

    print(f"Rebalances: {len(holdings)}, avg universe size at rebalance: "
          f"{holdings['n_assets'].mean():.1f}\n" if len(holdings) else "No rebalances.\n")

    print("=== Full period ===")
    report("long_short (top30% vs bottom30%)", ls_returns)
    report("long_only (top30%)", lo_returns)

    print(f"\n=== Walk-forward split at {TRAIN_END} — long_short ===")
    report("train", ls_returns[ls_returns.index < TRAIN_END])
    report("test (out-of-sample)", ls_returns[ls_returns.index >= TRAIN_END])

    print(f"\n=== Walk-forward split at {TRAIN_END} — long_only ===")
    report("train", lo_returns[lo_returns.index < TRAIN_END])
    report("test (out-of-sample)", lo_returns[lo_returns.index >= TRAIN_END])

    out = pd.DataFrame({"long_short": ls_returns, "long_only": lo_returns})
    out.to_csv("results/cross_sectional_momentum_returns.csv")
    holdings.to_csv("results/cross_sectional_momentum_holdings.csv", index=False)
    print("\nSaved to results/cross_sectional_momentum_returns.csv and "
          "results/cross_sectional_momentum_holdings.csv")


if __name__ == "__main__":
    main()
