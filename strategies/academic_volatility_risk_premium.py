"""Volatility risk premium harvesting (adapted, no options data).

Source: the volatility/variance risk premium is documented in e.g. Carr, P.,
Wu, L. (2009), "Variance Risk Premia", Review of Financial Studies 22(3),
1311-1341. The paperswithbacktest.com / Quantpedia strategy page
(https://quantpedia.com/strategies/volatility-risk-premium-effect/) harvests
it literally, by selling monthly at-the-money index straddles. Spot OHLCV
crypto data has no options chain, so this reimplements the *economic*
premise (implied/expected vol is usually priced above what subsequently
realizes, so being "long the market, short volatility" pays off in calm
regimes and gets hurt in vol spikes) via a realized-vol regime proxy: full
exposure when current vol is below its own longer-run average, cut back
when vol is spiking. No code reused, only the documented mechanic.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import pandas as pd

META = {
    "name": "academic_volatility_risk_premium",
    "category": "trend_following",
    "source": "Carr & Wu (2009) variance risk premium / Quantpedia 'Volatility Risk Premium Effect' (realized-vol proxy, no options data)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Compares short-window realized vol to a longer-window realized "
        "vol. When short vol < long vol (calm regime, the premium is safe "
        "to harvest) exposure is scaled up toward full long; when short vol "
        "spikes above the long-run average (crisis regime) exposure is cut "
        "toward flat."
    ),
    "default_params": {"short_window": 10, "long_window": 60},
}


def signals(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 60,
    **_,
) -> pd.Series:
    bar_return = df["close"].pct_change()
    vol_short = bar_return.rolling(short_window).std()
    vol_long = bar_return.rolling(long_window).std()

    weight = (vol_long / vol_short).clip(0.0, 1.0)
    return weight.fillna(0.0)
