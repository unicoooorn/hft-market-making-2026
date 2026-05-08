from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from backtester.core.interfaces import Executor
from backtester.model import (
    TradeEvent,
    OrderBookSnapshot,
    OrderIntent,
    ExecutionReport,
    ExecutionStatus,
    Side,
    PriceScaled,
    AmountScaled,
)

SCALE = 10**8


@dataclass
class LimitOrder:
    """Single active limit order (either bid or ask)."""
    order_id: UUID
    side: Side
    price: PriceScaled
    remaining_amount: AmountScaled


class LimitOrderExecutor(Executor):
    """Limit order executor with single bid/ask tracking."""

    def __init__(self) -> None:
        self._bid: Optional[LimitOrder] = None
        self._ask: Optional[LimitOrder] = None
        self._cancelled_count: int = 0

    def submit_order(self, order: OrderIntent):
        """Submit order with cancel-and-replace semantics."""
        side = Side.BUY if order.side == "buy" else Side.SELL

        new_order = LimitOrder(
            order_id=order.order_id,
            side=side,
            price=order.limit_price,
            remaining_amount=order.amount,
        )

        if side == Side.BUY:
            self._bid = new_order
        else:
            self._ask = new_order

        return

    def cancel(self) -> None:
        """Cancel all active orders."""
        self._bid = None
        self._ask = None
        self._cancelled_count += 1

    @property
    def cancelled_count(self) -> int:
        """Number of cancellations performed."""
        return self._cancelled_count

    def on_trade_event(self, event: TradeEvent) -> Optional[list[tuple[ExecutionReport, Side]]]:
        """Check if bid/ask fills against incoming trade."""
        reports: list[tuple[ExecutionReport, Side]] = []

        if self._bid is not None and event.price < self._bid.price:
            fill_amount = min(event.amount, self._bid.remaining_amount)
            if fill_amount > 0:
                self._bid.remaining_amount -= fill_amount
                status = (
                    ExecutionStatus.FILLED
                    if self._bid.remaining_amount == 0
                    else ExecutionStatus.PARTIALLY_FILLED
                )
                reports.append((ExecutionReport(
                    report_id=uuid4(),
                    order_id=self._bid.order_id,
                    status=status,
                    filled_price=event.price,
                    filled_amount=fill_amount,
                    timestamp=event.exchange_ts,
                    message=f"{'Filled' if status == ExecutionStatus.FILLED else 'Partial'}: {fill_amount/SCALE:.2f} @ {event.price/SCALE:.8f}",
                ), Side.BUY))
                if self._bid.remaining_amount == 0:
                    self._bid = None

        if self._ask is not None and event.price > self._ask.price:
            fill_amount = min(event.amount, self._ask.remaining_amount)
            if fill_amount > 0:
                self._ask.remaining_amount -= fill_amount
                status = (
                    ExecutionStatus.FILLED
                    if self._ask.remaining_amount == 0
                    else ExecutionStatus.PARTIALLY_FILLED
                )
                reports.append((ExecutionReport(
                    report_id=uuid4(),
                    order_id=self._ask.order_id,
                    status=status,
                    filled_price=event.price,
                    filled_amount=fill_amount,
                    timestamp=event.exchange_ts,
                    message=f"{'Filled' if status == ExecutionStatus.FILLED else 'Partial'}: {fill_amount/SCALE:.2f} @ {event.price/SCALE:.8f}",
                ), Side.SELL))
                if self._ask.remaining_amount == 0:
                    self._ask = None

        return reports if reports else None
