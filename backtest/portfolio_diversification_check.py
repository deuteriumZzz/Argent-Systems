"""Does combining low-correlation strategies clear the significance bar
that no individual strategy cleared in deflated_sharpe.py?

Portfolio theory says yes, if pairwise correlation is genuinely low: an
equal-weighted blend's Sharpe grows roughly with sqrt(N) for N
uncorrelated bets of similar quality (Grinold & Kahn's "fundamental law of
active management": IR ~ IC * sqrt(breadth)). The 5 candidates below were
picked BEFORE looking at correlations, one from each mechanism this
project's data pipelines support (price trend, calendar effect, on-chain,
funding carry, cross-exchange), specifically because different underlying
data sources should decorrelate better than five indicators computed off
the same OHLCV series.

Usage:
    python backtest/portfolio_diversification_check.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy.stats import norm

from data.cross_exchange import fetch_cross_exchange_premium
from data.fetch import fetch_ohlcv
from data.funding import fetch_funding_rate
from data.onchain import fetch_onchain_metrics
from strategies.registry import load_strategies

START = "2020-01-01"
FEE = 0.001
CARRY_FLIP_COST = 0.0008
EULER_MASCHERONI = 0.5772156649015329

CANDIDATES = [
    ("niche_dual_thrust", "ohlcv"),
    ("academic_turn_of_month_seasonality", "ohlcv"),
    ("onchain_hash_ribbon", "onchain"),
    ("carry_static_positive_funding", "funding_rate"),
    ("arbitrage_coinbase_premium_trend", "cross_exchange"),
]


def _price_returns(df: pd.DataFrame, strat) -> pd.Series:
    weight = strat.signals(df, **strat.meta["default_params"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pf = vbt.Portfolio.from_orders(
            close=df["close"], size=weight, size_type="targetpercent", fees=FEE, freq="1d"
        )
    return pf.returns()


def get_daily_returns(name: str, data_type: str, strategies: dict) -> pd.Series:
    strat = strategies[name]
    if data_type == "ohlcv":
        return _price_returns(fetch_ohlcv("BTC/USDT", "1d", START), strat)
    if data_type == "onchain":
        price_df = fetch_ohlcv("BTC/USDT", "1d", START)
        onchain_df = fetch_onchain_metrics(START)
        df = price_df.join(onchain_df.reindex(price_df.index, method="ffill"))
        df = df.dropna(subset=list(onchain_df.columns))
        return _price_returns(df, strat)
    if data_type == "cross_exchange":
        return _price_returns(fetch_cross_exchange_premium(START), strat)
    if data_type == "funding_rate":
        df = fetch_funding_rate("BTC/USDT:USDT", START)
        weight = strat.signals(df, **strat.meta["default_params"])
        position = weight.shift(1).fillna(0.0)
        period_return = position * df["funding_rate"] - position.diff().fillna(position).abs() * CARRY_FLIP_COST
        daily = (1 + period_return).groupby(period_return.index.normalize()).prod() - 1
        daily.index = pd.to_datetime(daily.index, utc=True)
        return daily
    raise ValueError(data_type)


def sharpe_stats(r: pd.Series, periods_per_year: int = 365) -> tuple[float, float, float, int]:
    mean, std = r.mean(), r.std()
    sharpe = (mean / std) * np.sqrt(periods_per_year) if std else float("nan")
    return sharpe, r.skew(), r.kurtosis() + 3, len(r)


def probabilistic_sharpe_ratio(sharpe_hat: float, sharpe_star: float, n: int, skew: float, kurt: float) -> float:
    numerator = (sharpe_hat - sharpe_star) * np.sqrt(n - 1)
    denominator = np.sqrt(max(1 - skew * sharpe_hat + ((kurt - 1) / 4) * sharpe_hat**2, 1e-12))
    return float(norm.cdf(numerator / denominator))


def sortino_ratio(r: pd.Series, periods_per_year: int = 365, target: float = 0.0) -> float:
    """Like Sharpe, but only penalizes downside deviation — doesn't treat a
    strategy's small steady gains as 'risk' just because they're not zero."""
    downside = (r[r < target] - target)
    downside_dev = np.sqrt((downside**2).mean()) if len(downside) else float("nan")
    if not downside_dev:
        return float("nan")
    return (r.mean() - target) / downside_dev * np.sqrt(periods_per_year)


def cvar(r: pd.Series, alpha: float = 0.95) -> float:
    """Expected return on the worst (1-alpha) fraction of days — the
    average size of a bad day, not just how often bad days happen."""
    threshold = r.quantile(1 - alpha)
    tail = r[r <= threshold]
    return float(tail.mean()) if len(tail) else float("nan")


def max_drawdown_pct(r: pd.Series) -> float:
    equity = (1 + r).cumprod()
    return float(((equity - equity.cummax()) / equity.cummax()).min() * 100)


def risk_summary(name: str, r: pd.Series) -> dict:
    sharpe, skew, kurt, n = sharpe_stats(r)
    return {
        "name": name,
        "sharpe": sharpe,
        "sortino": sortino_ratio(r),
        "cvar_95_pct": cvar(r) * 100,
        "max_drawdown_pct": max_drawdown_pct(r),
        "skew": skew,
        "kurt": kurt,
    }


def main() -> None:
    strategies = {s.meta["name"]: s for s in load_strategies()}

    returns = {name: get_daily_returns(name, dtype, strategies) for name, dtype in CANDIDATES}
    returns_df = pd.DataFrame(returns).dropna()

    print(f"Overlapping days across all 5: {len(returns_df)} ({returns_df.index.min().date()} - {returns_df.index.max().date()})\n")
    print("=== Pairwise correlation of daily returns ===")
    pd.set_option("display.width", 200)
    print(returns_df.corr().round(3).to_string())

    print("\n=== Individual Sharpe on this common window ===")
    individual = {}
    for name in returns_df.columns:
        s = sharpe_stats(returns_df[name])
        individual[name] = s
        print(f"  {name}: sharpe={s[0]:.3f}")

    portfolio_returns = returns_df.mean(axis=1)  # equal-weighted
    port_sharpe, port_skew, port_kurt, port_n = sharpe_stats(portfolio_returns)
    psr_vs_zero = probabilistic_sharpe_ratio(port_sharpe, 0.0, port_n, port_skew, port_kurt)

    print(f"\n=== Equal-weighted portfolio of all 5 ===")
    print(f"sharpe={port_sharpe:.3f}, skew={port_skew:.3f}, kurt={port_kurt:.3f}, n={port_n}")
    print(f"PSR vs benchmark=0 (single hypothesis, not corrected for the earlier 105-strategy search): {psr_vs_zero:.4f}")

    # --- Inverse-volatility (risk-parity) weighting ---
    # Weight day t by 1/rolling_vol computed from days *before* t (shifted),
    # so the allocation is knowable in advance, not fit on the same day's
    # return. Equal-weighting let carry's absurd Sharpe get diluted by the
    # noisier price-based legs; risk parity should instead lean into
    # whichever leg is quietest, which on this data means carry.
    vol_lookback = 30
    rolling_vol = returns_df.rolling(vol_lookback).std()
    inv_vol = 1 / rolling_vol.replace(0, np.nan)
    rp_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1)
    rp_returns = (returns_df * rp_weights).sum(axis=1, min_count=1).dropna()

    rp_sharpe, rp_skew, rp_kurt, rp_n = sharpe_stats(rp_returns)
    rp_psr = probabilistic_sharpe_ratio(rp_sharpe, 0.0, rp_n, rp_skew, rp_kurt)

    print(f"\n=== Risk-parity (inverse {vol_lookback}d-vol) portfolio of all 5 ===")
    print(f"sharpe={rp_sharpe:.3f}, skew={rp_skew:.3f}, kurt={rp_kurt:.3f}, n={rp_n}")
    print(f"PSR vs benchmark=0: {rp_psr:.4f}")
    print("\nAverage weight per strategy:")
    print(rp_weights.mean().round(3).to_string())

    out = returns_df.copy()
    out["portfolio_equal_weight"] = portfolio_returns
    out["portfolio_risk_parity"] = rp_returns
    out.to_csv("results/portfolio_diversification_returns.csv")
    print("\nSaved daily returns to results/portfolio_diversification_returns.csv")

    # --- Risk summary: Sharpe vs Sortino vs CVaR vs max drawdown ---
    # Sharpe penalizes upside and downside volatility equally, which
    # flatters a "steady small gains, rare big loss" payoff (exactly
    # carry's shape). Sortino only counts downside; CVaR shows how bad the
    # bad days actually are. If Sharpe looks great but CVaR/drawdown look
    # ugly, that's the tail risk Sharpe was hiding.
    summary_rows = [risk_summary(name, returns_df[name]) for name in returns_df.columns]
    summary_rows.append(risk_summary("portfolio_equal_weight", portfolio_returns))
    summary_rows.append(risk_summary("portfolio_risk_parity", rp_returns))
    summary = pd.DataFrame(summary_rows)

    print("\n=== Risk summary: Sharpe vs Sortino vs CVaR(95%) vs max drawdown ===")
    print(summary.to_string(index=False))
    summary.to_csv("results/risk_summary.csv", index=False)
    print("\nSaved to results/risk_summary.csv")


if __name__ == "__main__":
    main()
