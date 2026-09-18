"""Generate the static PNG charts embedded in README.md.

GitHub renders README images as static files (no JS), so these are plain
matplotlib exports rather than the interactive HTML the dataviz skill
otherwise favors — but the same form/color rules apply: one hue per job
(sequential for magnitude, diverging for polarity, categorical for
identity), no dual axes, a legend for 2+ series. Colors are the validated
default palette (see the dataviz skill's references/palette.md) — light
chart surface, so they read cleanly regardless of the viewer's GitHub
theme.

Usage:
    python docs/generate_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# --- validated palette (dataviz skill references/palette.md) ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
DIVERGING_MID = "#f0efec"

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent / "charts"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "text.color": INK_PRIMARY,
        "axes.edgecolor": GRIDLINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def chart_category_breakdown() -> None:
    strategies_dir = Path(__file__).resolve().parent.parent / "strategies"
    files = sorted(p.stem for p in strategies_dir.glob("*.py") if p.stem != "registry")
    known_prefixes = [
        "academic", "trend", "reversion", "volume", "niche", "community",
        "nfi", "grid", "smc", "pattern", "carry", "onchain", "dca", "arbitrage", "custom",
    ]
    counts: dict[str, int] = {}
    for name in files:
        prefix = next((p for p in known_prefixes if name.startswith(p + "_")), "other (pilot)")
        counts[prefix] = counts.get(prefix, 0) + 1
    series = pd.Series(counts).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(series.index, series.values, color=BLUE, height=0.6)
    for i, v in enumerate(series.values):
        ax.text(v + 0.5, i, str(v), va="center", color=INK_SECONDARY, fontsize=10)
    ax.set_xlabel(f"Number of strategies (total: {series.sum()})")
    ax.set_title("Argent Systems — strategy count by source category", loc="left", fontsize=13, color=INK_PRIMARY, pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "category_breakdown.png", dpi=150)
    plt.close(fig)


def chart_regime_heatmap(top_n: int = 20) -> None:
    df = pd.read_csv(RESULTS / "regime_comparison_robustness.csv", index_col=0)
    regime_cols = [c for c in df.columns if c not in ("avg_sharpe", "regimes_positive")]
    top = df.sort_values(["regimes_positive", "avg_sharpe"], ascending=False).head(top_n)
    matrix = top[regime_cols].to_numpy()

    vmax = np.nanmax(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(11, 8))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("diverging", [RED, DIVERGING_MID, BLUE])
    im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(regime_cols)))
    ax.set_xticklabels(regime_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=9)
    ax.set_title(
        f"Sharpe by regime — top {top_n} strategies by robustness (11 BTC regimes, 2017-2025)",
        loc="left", fontsize=12, color=INK_PRIMARY, pad=12,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Sharpe ratio", color=INK_SECONDARY)
    cbar.ax.yaxis.set_tick_params(color=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "regime_heatmap.png", dpi=150)
    plt.close(fig)


def chart_deflated_sharpe() -> None:
    df = pd.read_csv(RESULTS / "deflated_sharpe_1h.csv")
    n_trials = len(df)
    var_sr = df["sharpe"].var(ddof=1)
    euler = 0.5772156649015329
    from scipy.stats import norm

    z1 = norm.ppf(1 - 1 / n_trials)
    z2 = norm.ppf(1 - 1 / (n_trials * np.e))
    benchmark = np.sqrt(var_sr) * ((1 - euler) * z1 + euler * z2)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["sharpe"], df["dsr"], s=28, color=BLUE, alpha=0.75, edgecolors="none")
    ax.axhline(0.95, color=RED, linewidth=1.5, linestyle="--")
    ax.text(df["sharpe"].min(), 0.965, "significance threshold (DSR = 0.95)", color=RED, fontsize=9)
    ax.axvline(benchmark, color=INK_MUTED, linewidth=1.2, linestyle=":")
    ax.text(benchmark, 0.05, f"  benchmark: expected max\n  Sharpe from {n_trials} trials of luck ({benchmark:.2f})",
            color=INK_SECONDARY, fontsize=8.5, va="bottom")
    ax.set_xlabel("Raw Sharpe ratio (continuous full-history backtest, hourly)")
    ax.set_ylabel("Deflated Sharpe Ratio (probability of real skill)")
    ax.set_title(
        f"Deflated Sharpe Ratio — {n_trials} strategies, 0 clear the significance bar",
        loc="left", fontsize=12, color=INK_PRIMARY, pad=12,
    )
    ax.set_ylim(-0.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "deflated_sharpe.png", dpi=150)
    plt.close(fig)


def chart_portfolio_diversification() -> None:
    df = pd.read_csv(RESULTS / "portfolio_diversification_returns.csv", index_col=0, parse_dates=True)
    series_to_plot = {
        "carry_static_positive_funding": ("Carry alone", BLUE),
        "portfolio_equal_weight": ("Equal-weighted portfolio (5 strategies)", ORANGE),
        "portfolio_risk_parity": ("Risk-parity portfolio (5 strategies)", AQUA),
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    for col, (label, color) in series_to_plot.items():
        equity = (1 + df[col]).cumprod()
        ax.plot(equity.index, equity.values, color=color, linewidth=2, label=label)

    ax.set_yscale("log")
    ax.set_ylabel("Equity (log scale, starting at 1.0)")
    ax.set_title(
        "Higher raw return, lower risk-adjusted return: equal weight ends highest\n"
        "but both blends trail carry alone on Sharpe (0.87 / 3.24 vs 10.59)",
        loc="left", fontsize=12, color=INK_PRIMARY, pad=12,
    )
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "portfolio_diversification.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    chart_category_breakdown()
    chart_regime_heatmap()
    chart_deflated_sharpe()
    chart_portfolio_diversification()
    print(f"Saved 4 charts to {OUT}")
