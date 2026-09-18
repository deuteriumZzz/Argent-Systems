"""Halloween Indicator / "Sell in May and Go Away".

Source: Bouman, S. & Jacobsen, B. (2002), "The Halloween Indicator, 'Sell in
May and Go Away': Another Puzzle", American Economic Review 92(5), 1618-1635 -
finds statistically significant higher average equity returns during
November-April ("winter") than May-October ("summer") across most of the 37
countries studied. Purely calendar-based month-of-year seasonality; applies
directly to any single asset's time series with no adaptation needed beyond
using calendar months instead of an exchange trading calendar. No code
reused, only the documented rule.
License: N/A (published academic strategy, reimplemented from the paper's rule).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

META = {
    "name": "academic_halloween_indicator",
    "category": "seasonality",
    "source": "Bouman & Jacobsen (2002), 'The Halloween Indicator', American Economic Review 92(5)",
    "license": "N/A (published academic strategy, reimplemented from the paper's rule)",
    "description": (
        "Long during the November-April 'winter' half of the year, flat "
        "(or short, if `allow_short`) during the May-October 'summer' half."
    ),
    "default_params": {"allow_short": False},
}


def signals(df: pd.DataFrame, allow_short: bool = False, **_) -> pd.Series:
    month = df.index.month
    in_winter_half = np.isin(month, [11, 12, 1, 2, 3, 4])
    weight = pd.Series(
        np.where(in_winter_half, 1.0, (-1.0 if allow_short else 0.0)),
        index=df.index,
    )
    return weight
