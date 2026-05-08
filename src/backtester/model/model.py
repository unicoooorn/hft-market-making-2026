from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from uuid import UUID

# Use raw int64 microseconds internally - convert to datetime only when needed
TimestampUs = int  # Microseconds since epoch
Symbol = str

# Price and amount are stored as scaled integers (value * 1e8)
PriceScaled = int
AmountScaled = int


class Side(Enum):
    SELL = 0
    BUY = 1


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """Trade/tick event using scaled integers and raw timestamp."""
    trade_seq: int
    timestamp_us: TimestampUs
    side: Side
    price: PriceScaled
    amount: AmountScaled
    symbol: Symbol = "CMF"
    
    @property
    def exchange_ts(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_us / 1_000_000)
    
    @property
    def price_float(self) -> float:
        return self.price / 1e8
    
    @property
    def amount_float(self) -> float:
        return self.amount / 1e8


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """Top-of-book snapshot - only best bid/ask with volumes.
    
    Memory per snapshot:
    - Old: 50 OrderBookLevel objects = ~4KB+
    - With flat tuples: 50 tuples = ~400 bytes
    - Now: 4 int64 values = ~32 bytes (125x reduction vs original!)
    """
    snapshot_seq: int
    timestamp_us: TimestampUs
    # Top of book only: price and amount for best bid/ask
    bid_price: PriceScaled
    bid_amount: AmountScaled
    ask_price: PriceScaled
    ask_amount: AmountScaled
    symbol: Symbol = "CMF"
    
    @property
    def exchange_ts(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_us / 1_000_000)
    
    @property
    def best_bid_price(self) -> float | None:
        return self.bid_price / 1e8 if self.bid_price > 0 else None
    
    @property
    def best_ask_price(self) -> float | None:
        return self.ask_price / 1e8 if self.ask_price > 0 else None
    
    @property
    def mid_price(self) -> float | None:
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2e8
        return None
    
    @property
    def spread(self) -> float | None:
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.ask_price - self.bid_price) / 1e8
        return None


@dataclass(frozen=True, slots=True)
class OrderIntent:
    order_id: UUID
    symbol: Symbol
    side: str
    order_type: str
    amount: AmountScaled
    limit_price: PriceScaled | None
    created_ts: datetime
    time_in_force: str = "GTC"
    cancel_and_replace: bool = True


class ExecutionStatus(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FILLED = "filled"
    CANCELLED = "cancelled"
    PARTIALLY_FILLED = "partially_filled"


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    report_id: UUID
    order_id: UUID
    status: ExecutionStatus
    filled_price: PriceScaled | None
    filled_amount: AmountScaled
    timestamp: datetime
    message: str | None = None
