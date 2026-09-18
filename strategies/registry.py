"""Auto-discovery for strategy modules.

Contract every file in this package (except this one and __init__) must
follow so it shows up in backtests automatically:

    META: dict with at least "name", "category", "source", "license",
          "description", "default_params".
    def signals(df: pd.DataFrame, **params) -> pd.Series:
        Target position weight per bar, aligned to df.index.
        1.0 = fully long, 0.0 = flat, -1.0 = fully short (spot-only
        strategies just never emit negative weight).

Drop a new file in strategies/ and it is picked up on the next run — no
manual registration.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

STRATEGIES_DIR = Path(__file__).parent
_SKIP = {"__init__", "registry"}


@dataclass
class Strategy:
    meta: dict
    signals: Callable[..., pd.Series]


def load_strategies() -> list[Strategy]:
    strategies = []
    for path in sorted(STRATEGIES_DIR.glob("*.py")):
        if path.stem in _SKIP:
            continue
        spec = importlib.util.spec_from_file_location(f"strategies.{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        if hasattr(module, "META") and hasattr(module, "signals"):
            strategies.append(Strategy(meta=module.META, signals=module.signals))
    return strategies
