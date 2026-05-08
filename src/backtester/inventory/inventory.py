from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from backtester.model import (
    ExecutionReport,
    ExecutionStatus,
    Side,
    PriceScaled,
    TimestampUs,
)

SCALE = 10**8


@dataclass(frozen=True, slots=True)
class InventorySnapshot:
    """Point-in-time snapshot of inventory state for charting."""
    timestamp_us: TimestampUs
    usd_balance: int
    token_balance: int
    equity: int
    pnl: int
    turnover: int
    mark_price: PriceScaled


class Inventory:
    """Tracks balances and PnL for the strategy."""

    def __init__(self, initial_usd: int = 0, initial_tokens: int = 0) -> None:
        self._usd_balance: int = initial_usd
        self._token_balance: int = initial_tokens
        self._mark_price: PriceScaled = 0
        self._history: list[InventorySnapshot] = []
        self._initial_equity: int = initial_usd
        self._turnover: int = 0

    def process_fill(
        self,
        report: ExecutionReport,
        side: Side,
        event_timestamp_us: TimestampUs,
    ) -> None:
        """Process execution report and update balances."""
        if report.status not in (
            ExecutionStatus.FILLED,
            ExecutionStatus.PARTIALLY_FILLED,
        ):
            return

        if report.filled_price is None or report.filled_amount == 0:
            return

        filled_amount = report.filled_amount
        filled_price = report.filled_price
        fill_value_usd = (filled_amount * filled_price) // SCALE

        self._turnover += fill_value_usd

        if side == Side.BUY:
            self._token_balance += filled_amount
            self._usd_balance -= fill_value_usd
        else:
            self._token_balance -= filled_amount
            self._usd_balance += fill_value_usd

    @property
    def equity(self) -> int:
        """Current equity = USD + token_value_at_mark_price."""
        token_value = (self._token_balance * self._mark_price) // SCALE
        return self._usd_balance + token_value

    @property
    def token_balance(self) -> int:
        """Current token position (scaled by 1e8)."""
        return self._token_balance

    def update_mark_price(self, price: PriceScaled, timestamp_us: TimestampUs) -> None:
        """Update mark price and record snapshot."""
        self._mark_price = price
        self._record_snapshot(timestamp_us)

    def _record_snapshot(self, timestamp_us: TimestampUs) -> None:
        """Record current state for charting."""
        equity = self._usd_balance + (self._token_balance * self._mark_price) // SCALE
        snapshot = InventorySnapshot(
            timestamp_us=timestamp_us,
            usd_balance=self._usd_balance,
            token_balance=self._token_balance,
            equity=equity,
            pnl=equity - self._initial_equity,
            turnover=self._turnover,
            mark_price=self._mark_price,
        )
        self._history.append(snapshot)

    def get_chart_data(self) -> dict[str, list]:
        """Get chart data for equity, PnL, USD, tokens, and turnover."""
        if not self._history:
            return {
                "timestamps": [],
                "equity": [],
                "pnl": [],
                "usd": [],
                "tokens": [],
                "turnover": [],
            }

        return {
            "timestamps": [
                datetime.fromtimestamp(s.timestamp_us / 1_000_000)
                for s in self._history
            ],
            "equity": [s.equity / SCALE for s in self._history],
            "pnl": [s.pnl / SCALE for s in self._history],
            "usd": [s.usd_balance / SCALE for s in self._history],
            "tokens": [s.token_balance / SCALE for s in self._history],
            "turnover": [s.turnover / SCALE for s in self._history],
        }
