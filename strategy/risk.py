"""
Risk management layer.

This is what stands between "a signal fired" and "we bet the account on it."
Three independent controls, all deliberately conservative defaults:

1. Volatility targeting -- size positions so each position contributes
   roughly the same amount of risk, instead of using a flat dollar amount.
2. Per-trade stop loss -- hard cap on loss for any single position.
3. Portfolio-level kill switch -- if cumulative drawdown breaches a limit,
   flatten everything and stop trading until reset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskLimits:
    target_annual_vol: float = 0.15      # size positions to target ~15% annualized vol
    max_position_weight: float = 0.25    # never put more than 25% of capital in one name
    stop_loss_pct: float = 0.08          # exit a position if it's down 8% from entry
    max_portfolio_drawdown: float = 0.20  # kill switch: stop trading at -20% from peak


def volatility_target_weight(
    returns: pd.Series,
    target_annual_vol: float,
    lookback: int = 20,
    trading_days: int = 252,
) -> pd.Series:
    """Scales a raw [-1, 1] signal into a capital weight based on realized volatility.

    Lower realized vol -> larger position (up to max_position_weight);
    higher realized vol -> smaller position. This is the standard vol-targeting
    approach discussed in Quantitative Equity Portfolio Management.
    """
    realized_vol = returns.rolling(lookback).std() * np.sqrt(trading_days)
    realized_vol = realized_vol.replace(0, np.nan)
    scale = (target_annual_vol / realized_vol).clip(upper=1.0)
    return scale.fillna(0.0)


def apply_position_limits(weight: pd.Series, limits: RiskLimits) -> pd.Series:
    return weight.clip(-limits.max_position_weight, limits.max_position_weight)


def apply_stop_loss(
    price: pd.Series,
    positions: pd.Series,
    limits: RiskLimits,
) -> pd.Series:
    """Zeroes out a position once it has drawn down more than stop_loss_pct
    from the price at which the position was entered.
    """
    adjusted = positions.copy()
    entry_price = None
    current_side = 0

    for i in range(len(positions)):
        pos = positions.iloc[i]
        px = price.iloc[i]

        side = np.sign(pos)
        if side != current_side:
            entry_price = px
            current_side = side

        if entry_price is not None and current_side != 0:
            adverse_move = (px - entry_price) / entry_price * current_side
            if adverse_move < -limits.stop_loss_pct:
                adjusted.iloc[i] = 0.0
                current_side = 0
                entry_price = None

    return adjusted


def check_kill_switch(equity_curve: pd.Series, limits: RiskLimits) -> bool:
    """Returns True if the portfolio should stop trading (drawdown breached)."""
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return bool((drawdown < -limits.max_portfolio_drawdown).any())


def size_position(
    raw_signal: pd.Series,
    price: pd.Series,
    limits: RiskLimits = RiskLimits(),
) -> pd.Series:
    """Full pipeline: raw signal -> vol-targeted weight -> position-limited ->
    stop-loss applied. This is what should feed into the backtest/execution
    layer, not the raw signal from strategy/signals.py directly.
    """
    returns = price.pct_change().fillna(0.0)
    vol_scale = volatility_target_weight(returns, limits.target_annual_vol)
    weighted = raw_signal * vol_scale
    limited = apply_position_limits(weighted, limits)
    final = apply_stop_loss(price, limited, limits)
    return final
