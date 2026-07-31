"""
Vectorized backtest engine.

Deliberately simple and transparent (no black-box framework) so you can see
exactly how positions turn into returns. Includes the standard gotchas:
- signal is lagged by 1 day before being applied (no lookahead bias)
- transaction costs are charged on position changes
- performance metrics follow standard definitions (annualized Sharpe, CAGR,
  max drawdown) -- cross-check these against Advances in Financial ML's
  discussion of backtest overfitting before trusting a "good" Sharpe.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    sharpe: float
    cagr: float
    max_drawdown: float
    win_rate: float
    n_trades: int


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """
    df: OHLCV dataframe with a 'Close' column, datetime index
    signal: position series in [-1, 1], same index as df
    transaction_cost_bps: cost per unit of position change, in basis points
    """
    price = df["Close"]
    daily_returns = price.pct_change().fillna(0.0)

    # lag signal by 1 day: today's position was decided using yesterday's close
    positions = signal.shift(1).fillna(0.0)

    # transaction costs charged on position changes (turnover)
    turnover = positions.diff().abs().fillna(positions.abs())
    costs = turnover * (transaction_cost_bps / 10_000)

    strategy_returns = positions * daily_returns - costs
    equity_curve = (1 + strategy_returns).cumprod()

    n_years = len(df) / TRADING_DAYS_PER_YEAR
    cagr = equity_curve.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else 0.0

    ann_vol = strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (strategy_returns.mean() * TRADING_DAYS_PER_YEAR) / ann_vol if ann_vol > 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    trade_days = strategy_returns[turnover > 0]
    win_rate = (trade_days > 0).mean() if len(trade_days) > 0 else 0.0
    n_trades = int((turnover > 0).sum())

    return BacktestResult(
        equity_curve=equity_curve,
        returns=strategy_returns,
        positions=positions,
        sharpe=sharpe,
        cagr=cagr,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        n_trades=n_trades,
    )


def print_report(ticker: str, signal_name: str, result: BacktestResult) -> None:
    print(f"\n=== Backtest: {ticker} / {signal_name} ===")
    print(f"CAGR:          {result.cagr:.2%}")
    print(f"Sharpe ratio:  {result.sharpe:.2f}")
    print(f"Max drawdown:  {result.max_drawdown:.2%}")
    print(f"Win rate:      {result.win_rate:.2%}")
    print(f"# of trades:   {result.n_trades}")


def plot_equity_curve(ticker: str, signal_name: str, result: BacktestResult) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    result.equity_curve.plot(ax=ax, label="Strategy")
    ax.set_title(f"{ticker} — {signal_name} equity curve")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"{ticker}_{signal_name}_equity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
