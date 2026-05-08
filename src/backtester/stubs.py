from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from backtester.core.interfaces import Strategy
from backtester.model import (
    TradeEvent,
    OrderBookSnapshot,
    OrderIntent,
    Side,
)


class StubStrategy(Strategy):
    """Stub strategy that places orders at best bid/ask prices."""

    def __init__(self, strategy_id: str = "stub", order_volume: int = 1000) -> None:
        self.strategy_id = strategy_id
        self._order_volume = order_volume

    def on_order_book_snapshot(
        self,
        snapshot: OrderBookSnapshot,
    ) -> list[OrderIntent]:
        """Place orders at best bid and ask prices."""
        intents: list[OrderIntent] = []
        now = datetime.now()

        if snapshot.bid_price > 0:
            intents.append(OrderIntent(
                order_id=uuid4(),
                symbol=snapshot.symbol,
                side=Side.BUY.name.lower(),
                order_type="limit",
                amount=self._order_volume,
                limit_price=snapshot.bid_price,
                created_ts=now,
            ))

        if snapshot.ask_price > 0:
            intents.append(OrderIntent(
                order_id=uuid4(),
                symbol=snapshot.symbol,
                side=Side.SELL.name.lower(),
                order_type="limit",
                amount=self._order_volume,
                limit_price=snapshot.ask_price,
                created_ts=now,
            ))

        return intents

    def on_trade_event(
        self,
        event: TradeEvent,
    ) -> list[OrderIntent]:
        """Stub implementation - no orders on trades."""
        return []
