"""
Simulated paper-trading loop.

Deliberately has NO broker connection. It replays your signal + risk logic
day by day over historical (or, if you extend it, freshly-fetched) data,
logs simulated fills and equity to CSV, so you can evaluate behavior over
time before ever wiring up a real broker API.

When you're ready for live execution, the intended extension point is
`SimulatedBroker.submit_order` -- replace that one method with a real
broker adapter (e.g. Alpaca) and everything upstream (signal generation,
risk sizing, logging) stays the same.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OUTPUT_DIR
from data.fetch_data import load_cached
from strategy.risk import RiskLimits, check_kill_switch, size_position
from strategy.signals import get_signal


@dataclass
class SimulatedBroker:
    cash: float
    holdings: dict[str, float] = field(default_factory=dict)  # ticker -> shares
    fill_log: list[dict] = field(default_factory=list)

    def submit_order(self, date, ticker: str, target_weight: float, price: float, equity: float):
        """Rebalances `ticker` to `target_weight` fraction of current equity.

        This is simulated: fills happen instantly at `price` with no slippage
        modeled beyond what's already charged as transaction cost upstream.
        Replace this method with a real broker call for live trading.
        """
        target_value = target_weight * equity
        target_shares = target_value / price if price > 0 else 0.0
        current_shares = self.holdings.get(ticker, 0.0)
        delta_shares = target_shares - current_shares

        self.cash -= delta_shares * price
        self.holdings[ticker] = target_shares

        self.fill_log.append(
            {
                "date": date,
                "ticker": ticker,
                "action": "buy" if delta_shares > 0 else "sell" if delta_shares < 0 else "hold",
                "shares_delta": delta_shares,
                "price": price,
                "target_weight": target_weight,
                "cash_after": self.cash,
            }
        )

    def portfolio_value(self, prices: dict[str, float]) -> float:
        holdings_value = sum(
            shares * prices.get(ticker, 0.0) for ticker, shares in self.holdings.items()
        )
        return self.cash + holdings_value


def run_paper_trading(
    tickers: list[str],
    signal_name: str,
    initial_capital: float = 100_000.0,
    risk_limits: RiskLimits | None = None,
) -> pd.DataFrame:
    limits = risk_limits or RiskLimits()
    signal_fn = get_signal(signal_name)

    price_data = {t: load_cached(t) for t in tickers}
    common_index = price_data[tickers[0]].index
    for t in tickers[1:]:
        common_index = common_index.intersection(price_data[t].index)
    common_index = common_index.sort_values()

    sized_positions = {}
    for t in tickers:
        df = price_data[t].loc[common_index]
        raw_signal = signal_fn(df)
        sized_positions[t] = size_position(raw_signal, df["Close"], limits)

    broker = SimulatedBroker(cash=initial_capital)
    equity_history = []
    trading_halted = False

    for date in common_index:
        prices_today = {t: price_data[t].loc[date, "Close"] for t in tickers}
        equity = broker.portfolio_value(prices_today)

        if not trading_halted:
            equity_curve_so_far = pd.Series([e["equity"] for e in equity_history] + [equity])
            if len(equity_curve_so_far) > 5 and check_kill_switch(equity_curve_so_far, limits):
                trading_halted = True
                print(f"[{date.date()}] Kill switch triggered — drawdown limit breached. "
                      f"Flattening and halting.")
                for t in tickers:
                    broker.submit_order(date, t, 0.0, prices_today[t], equity)

        if not trading_halted:
            for t in tickers:
                target_weight = sized_positions[t].loc[date]
                broker.submit_order(date, t, target_weight, prices_today[t], equity)

        equity_history.append({"date": date, "equity": broker.portfolio_value(prices_today)})

    equity_df = pd.DataFrame(equity_history).set_index("date")
    fills_df = pd.DataFrame(broker.fill_log)

    equity_path = OUTPUT_DIR / "paper_trading_equity.csv"
    fills_path = OUTPUT_DIR / "paper_trading_log.csv"
    equity_df.to_csv(equity_path)
    fills_df.to_csv(fills_path, index=False)

    final_equity = equity_df["equity"].iloc[-1]
    total_return = final_equity / initial_capital - 1
    print(f"\nPaper trading complete: {tickers} / {signal_name}")
    print(f"Final equity: ${final_equity:,.2f}  ({total_return:+.2%})")
    print(f"Fill log:    {fills_path}")
    print(f"Equity log:  {equity_path}")

    return equity_df
