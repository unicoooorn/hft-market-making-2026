from __future__ import annotations

from dataclasses import dataclass

from backtester.core.interfaces import (
    DataProvider,
    Strategy,
    Executor,
    BacktesterCore,
)
from backtester.inventory import Inventory
from backtester.model import (
    TradeEvent,
    OrderBookSnapshot,
)


@dataclass
class QuoteSnapshot:
    """Snapshot of market and strategy quotes."""
    timestamp_us: int
    best_bid: float
    best_ask: float
    strategy_bid: float
    strategy_ask: float
    mid_price: float


class Backtester(BacktesterCore):
    """Backtester with dependency injection."""

    def __init__(
        self,
        data_provider: DataProvider,
        strategy: Strategy,
        executor: Executor,
        inventory: Inventory | None = None,
    ) -> None:
        self._data_provider = data_provider
        self._strategy = strategy
        self._executor = executor
        self._inventory = inventory or Inventory()
        self._quote_history: list[QuoteSnapshot] = []

    @property
    def quote_history(self) -> list[QuoteSnapshot]:
        """History of market and strategy quotes for analysis."""
        return self._quote_history.copy()

    def run(self) -> None:
        """Run the backtest simulation loop."""
        while True:
            event = self._data_provider.next()
            if event is None:
                break

            if isinstance(event, OrderBookSnapshot):
                best_bid = event.bid_price / 1e8 if event.bid_price > 0 else 0
                best_ask = event.ask_price / 1e8 if event.ask_price > 0 else 0
                mid_price = (best_bid + best_ask) / 2

                order_intents = self._strategy.on_order_book_snapshot(event)

                strategy_bid = 0.0
                strategy_ask = 0.0
                for intent in order_intents:
                    if intent.limit_price is not None:
                        price = intent.limit_price / 1e8
                        if intent.side == "buy":
                            strategy_bid = price
                        elif intent.side == "sell":
                            strategy_ask = price
                    self._executor.submit_order(intent)

                if best_bid > 0 or best_ask > 0:
                    self._quote_history.append(QuoteSnapshot(
                        timestamp_us=event.timestamp_us,
                        best_bid=best_bid,
                        best_ask=best_ask,
                        strategy_bid=strategy_bid,
                        strategy_ask=strategy_ask,
                        mid_price=mid_price,
                    ))

                if event.mid_price is not None:
                    mid_price_scaled = int(event.mid_price * 1e8)
                    self._inventory.update_mark_price(mid_price_scaled, event.timestamp_us)

            elif isinstance(event, TradeEvent):
                fill_reports = self._executor.on_trade_event(event)
                if fill_reports:
                    for report, side in fill_reports:
                        self._inventory.process_fill(report, side, event.timestamp_us)

                order_intents = self._strategy.on_trade_event(event)
                for intent in order_intents:
                    if intent.side == "cancel" or intent.order_type == "cancel":
                        self._executor.cancel()
                    else:
                        self._executor.submit_order(intent)
