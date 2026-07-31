"""Fetches historical OHLCV data for a list of tickers and caches it as CSV."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR


def fetch_ticker(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check the symbol and date range.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    return df


def fetch_and_cache(tickers: list[str], start: str, end: str | None = None) -> dict[str, Path]:
    paths = {}
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = fetch_ticker(ticker, start, end)
        out_path = DATA_DIR / f"{ticker}.csv"
        df.to_csv(out_path)
        paths[ticker] = out_path
        print(f"  -> {len(df)} rows saved to {out_path}")
    return paths


def load_cached(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached data for {ticker}. Run: python cli.py fetch-data --tickers {ticker}"
        )
    return pd.read_csv(path, index_col="date", parse_dates=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and cache historical OHLCV data")
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    fetch_and_cache(args.tickers, args.start, args.end)
