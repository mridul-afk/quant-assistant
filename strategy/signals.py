"""
Signal generators.

Every function here has the signature: price_df -> position_series
where position_series is in [-1, 1] (fraction of capital to be long/short,
before risk-based sizing is applied in risk.py).

These are intentionally simple, well-known signals to give you a working
skeleton -- read Advances in Financial Machine Learning and Quantitative
Equity Portfolio Management for the real thing (meta-labeling, fractional
differentiation, factor models) and implement those here following the
same signature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_REGISTRY: dict[str, callable] = {}


def register(name: str):
    def decorator(fn):
        SIGNAL_REGISTRY[name] = fn
        return fn

    return decorator


@register("momentum")
def momentum_signal(df: pd.DataFrame, lookback: int = 60, threshold: float = 0.0) -> pd.Series:
    """Long if trailing return over `lookback` days is positive, short if negative."""
    returns = df["Close"].pct_change(lookback)
    position = np.sign(returns - threshold)
    return position.fillna(0.0).clip(-1, 1)


@register("mean_reversion")
def mean_reversion_signal(df: pd.DataFrame, window: int = 20, z_entry: float = 1.0) -> pd.Series:
    """Z-score of price vs rolling mean; fade extremes (short when high, long when low)."""
    rolling_mean = df["Close"].rolling(window).mean()
    rolling_std = df["Close"].rolling(window).std()
    zscore = (df["Close"] - rolling_mean) / rolling_std.replace(0, np.nan)

    position = pd.Series(0.0, index=df.index)
    position[zscore > z_entry] = -1.0
    position[zscore < -z_entry] = 1.0
    return position.fillna(0.0)


@register("dual_moving_average")
def dual_moving_average_signal(df: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    """Classic trend-following crossover: long when fast MA > slow MA, else short."""
    fast_ma = df["Close"].rolling(fast).mean()
    slow_ma = df["Close"].rolling(slow).mean()
    position = np.sign(fast_ma - slow_ma)
    return position.fillna(0.0).clip(-1, 1)


def get_signal(name: str) -> callable:
    if name not in SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal '{name}'. Available: {list(SIGNAL_REGISTRY)}")
    return SIGNAL_REGISTRY[name]
