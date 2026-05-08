from abc import ABC, abstractmethod

from backtester.model import (
    TradeEvent,
    OrderBookSnapshot,
    OrderIntent,
    ExecutionReport,
)


class DataProvider(ABC):
    """Abstract base class for data provider.
    
    Provides a stream of market data events (either TradeEvent or OrderBookSnapshot).
    """

    @abstractmethod
    def next(self) -> TradeEvent | OrderBookSnapshot | None:
        """Get next market data event.
        
        Returns:
            TradeEvent, OrderBookSnapshot, or None if no more data.
        """
        raise NotImplementedError


class Strategy(ABC):
    """Abstract base class for trading strategy."""

    strategy_id: str

    @abstractmethod
    def on_order_book_snapshot(
        self,
        snapshot: OrderBookSnapshot,
    ) -> list[OrderIntent]:
        """Handle order book snapshot and return order intents."""
        raise NotImplementedError

    @abstractmethod
    def on_trade_event(
        self,
        event: TradeEvent,
    ) -> list[OrderIntent]:
        """Handle trade event and return order intents."""
        raise NotImplementedError


class Executor(ABC):
    """Abstract base class for order executor."""

    @abstractmethod
    def submit_order(self, order: OrderIntent) -> ExecutionReport:
        """Submit order and return execution report."""
        raise NotImplementedError

    @abstractmethod
    def on_trade_event(
        self,
        event: TradeEvent,
    ) -> list[ExecutionReport] | None:
        """Process trade event and return execution reports if orders filled."""
        raise NotImplementedError


class BacktesterCore(ABC):
    """Abstract base class for backtester core."""

    @abstractmethod
    def run(self) -> None:
        """Run the backtest simulation."""
        raise NotImplementedError
