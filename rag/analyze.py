"""
Grounds an LLM answer in REAL computed metrics from your own backtest engine,
not book excerpts and not price prediction.

This deliberately does NOT predict future prices -- nothing in this project
can do that responsibly. What it does: runs every signal in strategy/signals.py
against a ticker's actual historical data, computes real Sharpe/CAGR/drawdown
via strategy/backtest.py, determines each signal's CURRENT state (long/flat/
short as of the most recent bar), and hands all of that to the LLM to explain
in plain language -- with citations to the actual numbers, not vibes.

The LLM's job here is explanation and synthesis of numbers that already
exist, not generation of new numbers. If it says "Sharpe is 0.8", that 0.8
came from run_backtest(), not from the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import ANTHROPIC_MODEL  # noqa: F401 (kept for parity with query.py imports)
from data.fetch_data import load_cached
from rag.llm_backend import get_backend
from strategy.backtest import run_backtest
from strategy.risk import RiskLimits, size_position
from strategy.signals import SIGNAL_REGISTRY, get_signal

ANALYZE_SYSTEM_PROMPT = """You are a quantitative analyst explaining REAL, \
already-computed backtest metrics for a stock -- you are not predicting future \
prices and you must not imply otherwise. Every number in the user message came \
from an actual backtest run against real historical data; use those numbers, \
do not invent new ones.

Structure your answer as:
Thesis -- one sentence on what the data shows, hedged appropriately
Evidence -- cite the specific Sharpe/CAGR/drawdown/current-signal numbers given
Risks -- name at least one real risk specific to what the numbers show (e.g. \
a high Sharpe over a short/lucky window, or all signals disagreeing, or a \
current signal that just flipped)
Position-sizing note -- do not tell the user how much to buy; point back to \
strategy/risk.py's volatility targeting instead

Never give an unconditional buy/sell recommendation. Never state or imply a \
future price target or return prediction -- these metrics describe the past, \
not the future. If the signals disagree with each other, say so explicitly \
rather than picking one to feature."""


def analyze_ticker(ticker: str, signal_names: list[str] | None = None) -> str:
    signal_names = signal_names or list(SIGNAL_REGISTRY.keys())

    df = load_cached(ticker)  # raises a clear error if not fetched yet
    latest_price = df["Close"].iloc[-1]
    latest_date = df.index[-1]

    metrics_lines = [f"Ticker: {ticker}", f"Latest price: {latest_price:.2f} (as of {latest_date.date()})", ""]

    for name in signal_names:
        signal_fn = get_signal(name)
        raw_signal = signal_fn(df)
        sized = size_position(raw_signal, df["Close"], RiskLimits())
        result = run_backtest(df, sized)

        current_raw = raw_signal.iloc[-1]
        current_state = "long" if current_raw > 0 else "short" if current_raw < 0 else "flat"
        current_weight = sized.iloc[-1]

        metrics_lines.append(f"[{name}]")
        metrics_lines.append(f"  Current signal state: {current_state} (raw={current_raw:.3f})")
        metrics_lines.append(f"  Current risk-sized position weight: {current_weight:.3f}")
        metrics_lines.append(f"  Backtested CAGR: {result.cagr:.2%}")
        metrics_lines.append(f"  Backtested Sharpe ratio: {result.sharpe:.2f}")
        metrics_lines.append(f"  Backtested max drawdown: {result.max_drawdown:.2%}")
        metrics_lines.append(f"  Backtested win rate: {result.win_rate:.2%}")
        metrics_lines.append(f"  Number of trades in backtest: {result.n_trades}")
        metrics_lines.append("")

    metrics_block = "\n".join(metrics_lines)

    backend = get_backend()
    answer = backend.generate(
        system=ANALYZE_SYSTEM_PROMPT,
        user=f"Computed backtest metrics:\n\n{metrics_block}\n\nExplain what this data shows.",
        max_tokens=1200,
    )

    return metrics_block + "\n---\n" + answer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explain a ticker's real backtest metrics via LLM")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--signals", nargs="+", default=None,
                         choices=list(SIGNAL_REGISTRY.keys()) if SIGNAL_REGISTRY else None)
    args = parser.parse_args()
    print(analyze_ticker(args.ticker, args.signals))
