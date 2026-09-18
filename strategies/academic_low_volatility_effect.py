"""Low-volatility anomaly as a single-asset vol-targeting overlay.

Source: Ang, A., Hodrick, R.J., Xing, Y., Zhang, X. (2006), "The
Cross-Section of Volatility and Expected Returns", Journal of Finance
61(1), 259-299. Also a strategy page on paperswithbacktest.com / Quantpedia
(https://quantpedia.com/strategies/low-volatility-factor-effect-in-stocks-long-only-version/),
which longs the lowest-volatility decile of a stock universe. There is no
cross-section for a single asset, so this reimplements the anomaly's
single-asset analogue used across the industry (e.g. Frazzini & Pedersen's
2014 "Betting Against Beta" risk-parity-style scaling): stay long always,
but size the position up in low-realized-vol regimes and down in
high-vol regimes, since low vol historically carries better risk-adjusted
(not necessarily higher raw) returns. No code reused, only the rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "academic_low_volatility_effect",
    "category": "trend_following",
    "source": "Ang, Hodrick, Xing & Zhang (2006), 'The Cross-Section of Volatility and Expected Returns', JF 61(1)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Always long; exposure = clip(target_vol / realized_vol, 0, 1) "
        "using trailing realized volatility over `vol_window`. Low-vol "
        "regimes get full exposure, high-vol regimes get scaled down - the "
        "single-asset analogue of the paper's low-vol-decile long-only "
        "portfolio."
    ),
    "default_params": {"vol_window": 30, "target_vol": 0.015},
}


def signals(
    df: pd.DataFrame,
    vol_window: int = 30,
    target_vol: float = 0.015,
    **_,
) -> pd.Series:
    bar_return = df["close"].pct_change()
    realized_vol = bar_return.rolling(vol_window).std()

    weight = (target_vol / realized_vol).clip(0.0, 1.0)
    return weight.fillna(0.0)
