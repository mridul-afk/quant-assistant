"""
Single entry point for the whole project.

    python cli.py ingest-books --pdf-dir /path/to/quantbooks
    python cli.py ask "your question"
    python cli.py fetch-data --tickers AAPL MSFT --start 2018-01-01
    python cli.py backtest --ticker AAPL --signal momentum
    python cli.py paper-trade --tickers AAPL MSFT --signal momentum --capital 100000
"""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_ingest_books(args):
    from rag.ingest import ingest_directory

    ingest_directory(args.pdf_dir)


def cmd_ask(args):
    from rag.query import ask

    print(ask(args.question))


def cmd_fetch_data(args):
    from data.fetch_data import fetch_and_cache

    fetch_and_cache(args.tickers, args.start, args.end)


def cmd_backtest(args):
    from data.fetch_data import load_cached
    from strategy.backtest import plot_equity_curve, print_report, run_backtest
    from strategy.risk import RiskLimits, size_position
    from strategy.signals import get_signal

    df = load_cached(args.ticker)
    signal_fn = get_signal(args.signal)
    raw_signal = signal_fn(df)

    if args.no_risk_sizing:
        final_signal = raw_signal
    else:
        final_signal = size_position(raw_signal, df["Close"], RiskLimits())

    result = run_backtest(df, final_signal, transaction_cost_bps=args.cost_bps)
    print_report(args.ticker, args.signal, result)

    plot_path = plot_equity_curve(args.ticker, args.signal, result)
    print(f"Equity curve saved to: {plot_path}")


def cmd_paper_trade(args):
    from paper_trading.simulator import run_paper_trading

    run_paper_trading(args.tickers, args.signal, initial_capital=args.capital)


def cmd_analyze(args):
    from rag.analyze import analyze_ticker

    print(analyze_ticker(args.ticker, args.signals))


def main():
    parser = argparse.ArgumentParser(description="Quant assistant: RAG + backtesting + paper trading")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest-books", help="Ingest PDFs into the RAG vector store")
    p_ingest.add_argument("--pdf-dir", required=True, type=Path)
    p_ingest.set_defaults(func=cmd_ingest_books)

    p_ask = sub.add_parser("ask", help="Ask a question grounded in your book library")
    p_ask.add_argument("question", type=str)
    p_ask.set_defaults(func=cmd_ask)

    p_fetch = sub.add_parser("fetch-data", help="Fetch and cache historical OHLCV data")
    p_fetch.add_argument("--tickers", nargs="+", required=True)
    p_fetch.add_argument("--start", default="2018-01-01")
    p_fetch.add_argument("--end", default=None)
    p_fetch.set_defaults(func=cmd_fetch_data)

    p_bt = sub.add_parser("backtest", help="Backtest a signal on a single ticker")
    p_bt.add_argument("--ticker", required=True)
    p_bt.add_argument("--signal", required=True, choices=["momentum", "mean_reversion", "dual_moving_average"])
    p_bt.add_argument("--cost-bps", type=float, default=5.0, help="Transaction cost in bps per trade")
    p_bt.add_argument("--no-risk-sizing", action="store_true", help="Use raw signal instead of vol-targeted sizing")
    p_bt.set_defaults(func=cmd_backtest)

    p_pt = sub.add_parser("paper-trade", help="Run the simulated paper-trading loop")
    p_pt.add_argument("--tickers", nargs="+", required=True)
    p_pt.add_argument("--signal", required=True, choices=["momentum", "mean_reversion", "dual_moving_average"])
    p_pt.add_argument("--capital", type=float, default=100_000.0)
    p_pt.set_defaults(func=cmd_paper_trade)

    p_an = sub.add_parser("analyze", help="LLM explains real computed backtest metrics for a ticker (not a prediction)")
    p_an.add_argument("--ticker", required=True)
    p_an.add_argument("--signals", nargs="+", default=None,
                       choices=["momentum", "mean_reversion", "dual_moving_average"])
    p_an.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
