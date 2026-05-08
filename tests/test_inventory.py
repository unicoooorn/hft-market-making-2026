"""Tests for the inventory module."""

from datetime import datetime
from uuid import uuid4

import pytest

from backtester.inventory import Inventory, InventorySnapshot
from backtester.model import (
    ExecutionReport,
    ExecutionStatus,
    Side,
)

SCALE = 10**8


class TestInventoryInitialization:
    """Test inventory initialization."""

    def test_default_initialization(self) -> None:
        """Test inventory with default zero balances."""
        inv = Inventory()
        assert inv._usd_balance == 0
        assert inv._token_balance == 0
        assert inv._initial_equity == 0
        assert inv._turnover == 0
        assert inv._history == []

    def test_custom_initialization(self) -> None:
        """Test inventory with custom initial balances."""
        inv = Inventory(initial_usd=1000 * SCALE, initial_tokens=50 * SCALE)
        assert inv._usd_balance == 1000 * SCALE
        assert inv._token_balance == 50 * SCALE


class TestProcessFill:
    """Test fill processing logic."""

    def test_buy_fill_increases_tokens_decreases_usd(self) -> None:
        """Test that BUY fill increases tokens and decreases USD."""
        inv = Inventory(initial_usd=10000 * SCALE, initial_tokens=0)
        
        report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=100 * SCALE,  # $100 per token
            filled_amount=10 * SCALE,  # 10 tokens
            timestamp=datetime.now(),
        )
        
        inv.process_fill(report, Side.BUY, 1000000)
        
        assert inv._token_balance == 10 * SCALE
        assert inv._usd_balance == 10000 * SCALE - 1000 * SCALE  # 10 * $100

    def test_sell_fill_decreases_tokens_increases_usd(self) -> None:
        """Test that SELL fill decreases tokens and increases USD."""
        inv = Inventory(initial_usd=0, initial_tokens=10 * SCALE)
        
        report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=100 * SCALE,
            filled_amount=5 * SCALE,
            timestamp=datetime.now(),
        )
        
        inv.process_fill(report, Side.SELL, 1000000)
        
        assert inv._token_balance == 5 * SCALE
        assert inv._usd_balance == 500 * SCALE  # 5 * $100

    def test_partial_fill(self) -> None:
        """Test partial fill processing."""
        inv = Inventory(initial_usd=10000 * SCALE)
        
        report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.PARTIALLY_FILLED,
            filled_price=50 * SCALE,
            filled_amount=2 * SCALE,
            timestamp=datetime.now(),
        )
        
        inv.process_fill(report, Side.BUY, 1000000)
        
        assert inv._token_balance == 2 * SCALE
        assert inv._usd_balance == 10000 * SCALE - 100 * SCALE

    def test_pending_fill_ignored(self) -> None:
        """Test that PENDING status fills are ignored."""
        inv = Inventory(initial_usd=10000 * SCALE)
        initial_usd = inv._usd_balance
        initial_tokens = inv._token_balance
        
        report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.PENDING,
            filled_price=100 * SCALE,
            filled_amount=10 * SCALE,
            timestamp=datetime.now(),
        )
        
        inv.process_fill(report, Side.BUY, 1000000)
        
        assert inv._usd_balance == initial_usd
        assert inv._token_balance == initial_tokens

    def test_none_price_fill_ignored(self) -> None:
        """Test that fills with None price are ignored."""
        inv = Inventory(initial_usd=10000 * SCALE)
        initial_usd = inv._usd_balance
        
        report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=None,
            filled_amount=10 * SCALE,
            timestamp=datetime.now(),
        )
        
        inv.process_fill(report, Side.BUY, 1000000)
        
        assert inv._usd_balance == initial_usd

    def test_zero_amount_fill_ignored(self) -> None:
        """Test that fills with zero amount are ignored."""
        inv = Inventory(initial_usd=10000 * SCALE)
        initial_usd = inv._usd_balance
        
        report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=100 * SCALE,
            filled_amount=0,
            timestamp=datetime.now(),
        )
        
        inv.process_fill(report, Side.BUY, 1000000)
        
        assert inv._usd_balance == initial_usd


class TestPnLCalculation:
    """Test PnL calculation logic."""

    def test_pnl_from_round_trip_trade(self) -> None:
        """Test PnL from buying and selling at different prices.
        
        Note: In this simple model, PnL tracks realized profit/loss.
        Buy 10 @ $100 = -$1000 USD, +10 tokens
        Sell 10 @ $120 = +$1200 USD, -10 tokens
        Net: +$200 USD, 0 tokens -> PnL = $200
        """
        inv = Inventory(initial_usd=10000 * SCALE)
        
        # Buy 10 tokens at $100
        buy_report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=100 * SCALE,
            filled_amount=10 * SCALE,
            timestamp=datetime.now(),
        )
        inv.process_fill(buy_report, Side.BUY, 1000000)
        
        # Sell 10 tokens at $120
        sell_report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=120 * SCALE,
            filled_amount=10 * SCALE,
            timestamp=datetime.now(),
        )
        inv.process_fill(sell_report, Side.SELL, 2000000)
        
        # Net USD change: -1000 + 1200 = +200
        # PnL = final_equity - initial_equity = (10200) - (10000) = 200
        assert inv._usd_balance == 10200 * SCALE
        assert inv._token_balance == 0

    def test_pnl_loss_from_round_trip_trade(self) -> None:
        """Test PnL loss from buying high and selling low."""
        inv = Inventory(initial_usd=10000 * SCALE)
        
        # Buy 10 tokens at $100
        buy_report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=100 * SCALE,
            filled_amount=10 * SCALE,
            timestamp=datetime.now(),
        )
        inv.process_fill(buy_report, Side.BUY, 1000000)
        
        # Sell 10 tokens at $80
        sell_report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=80 * SCALE,
            filled_amount=10 * SCALE,
            timestamp=datetime.now(),
        )
        inv.process_fill(sell_report, Side.SELL, 2000000)
        
        # Net USD change: -1000 + 800 = -200
        assert inv._usd_balance == 9800 * SCALE
        assert inv._token_balance == 0


class TestEquityCalculation:
    """Test equity calculation."""

    def test_equity_with_no_tokens(self) -> None:
        """Test equity when holding only USD."""
        inv = Inventory(initial_usd=5000 * SCALE)
        inv.update_mark_price(100 * SCALE, 1000000)
        
        assert inv.equity == 5000 * SCALE

    def test_equity_with_tokens(self) -> None:
        """Test equity with USD and tokens."""
        inv = Inventory(initial_usd=5000 * SCALE, initial_tokens=10 * SCALE)
        inv.update_mark_price(100 * SCALE, 1000000)
        
        # Equity = 5000 USD + 10 tokens * $100 = $6000
        assert inv.equity == 6000 * SCALE

    def test_equity_with_negative_tokens(self) -> None:
        """Test equity with short token position."""
        inv = Inventory(initial_usd=10000 * SCALE, initial_tokens=-10 * SCALE)
        inv.update_mark_price(100 * SCALE, 1000000)
        
        # Equity = 10000 USD + (-10 tokens * $100) = $9000
        assert inv.equity == 9000 * SCALE


class TestMarkPriceUpdate:
    """Test mark price updates and snapshot recording."""

    def test_update_mark_price_records_snapshot(self) -> None:
        """Test that updating mark price records a snapshot."""
        inv = Inventory()
        
        inv.update_mark_price(100 * SCALE, 1000000)
        
        assert len(inv._history) == 1
        assert inv._history[0].mark_price == 100 * SCALE
        assert inv._history[0].timestamp_us == 1000000

    def test_multiple_mark_price_updates(self) -> None:
        """Test multiple mark price updates create multiple snapshots."""
        inv = Inventory()
        
        inv.update_mark_price(100 * SCALE, 1000000)
        inv.update_mark_price(105 * SCALE, 2000000)
        inv.update_mark_price(110 * SCALE, 3000000)
        
        assert len(inv._history) == 3
        assert inv._history[0].mark_price == 100 * SCALE
        assert inv._history[1].mark_price == 105 * SCALE
        assert inv._history[2].mark_price == 110 * SCALE


class TestChart:
    """Test chart data generation."""

    def test_get_chart_data_empty(self) -> None:
        """Test chart data with no snapshots."""
        inv = Inventory()
        chart_data = inv.get_chart_data()

        assert chart_data == {
            "timestamps": [],
            "equity": [],
            "pnl": [],
            "usd": [],
            "tokens": [],
            "turnover": [],
        }

    def test_get_chart_data_with_snapshots(self) -> None:
        """Test chart data with snapshots."""
        inv = Inventory(initial_usd=10000 * SCALE)

        inv.update_mark_price(100 * SCALE, 1000000)
        inv.update_mark_price(105 * SCALE, 2000000)

        chart_data = inv.get_chart_data()

        assert len(chart_data["timestamps"]) == 2
        assert len(chart_data["equity"]) == 2
        assert len(chart_data["pnl"]) == 2
        assert len(chart_data["usd"]) == 2
        assert len(chart_data["tokens"]) == 2
        assert len(chart_data["turnover"]) == 2

        assert isinstance(chart_data["equity"][0], float)
        assert chart_data["usd"][0] == 10000.0

    def test_get_chart_data_includes_fills(self) -> None:
        """Test chart data reflects fills."""
        inv = Inventory(initial_usd=10000 * SCALE)

        buy_report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=100 * SCALE,
            filled_amount=10 * SCALE,
            timestamp=datetime.now(),
        )
        inv.process_fill(buy_report, Side.BUY, 1000000)

        inv.update_mark_price(100 * SCALE, 2000000)

        chart_data = inv.get_chart_data()

        assert chart_data["usd"][-1] == 9000.0
        assert chart_data["tokens"][-1] == 10.0
        assert chart_data["turnover"][-1] == 1000.0


class TestInventorySnapshot:
    """Test InventorySnapshot dataclass."""

    def test_snapshot_creation(self) -> None:
        """Test creating an inventory snapshot."""
        snapshot = InventorySnapshot(
            timestamp_us=1000000,
            usd_balance=5000 * SCALE,
            token_balance=10 * SCALE,
            equity=6000 * SCALE,
            pnl=100 * SCALE,
            turnover=500 * SCALE,
            mark_price=100 * SCALE,
        )

        assert snapshot.timestamp_us == 1000000
        assert snapshot.usd_balance == 5000 * SCALE
        assert snapshot.token_balance == 10 * SCALE
        assert snapshot.equity == 6000 * SCALE
        assert snapshot.pnl == 100 * SCALE
        assert snapshot.turnover == 500 * SCALE
        assert snapshot.mark_price == 100 * SCALE

    def test_snapshot_immutability(self) -> None:
        """Test that snapshots are immutable."""
        snapshot = InventorySnapshot(
            timestamp_us=1000000,
            usd_balance=5000 * SCALE,
            token_balance=10 * SCALE,
            equity=6000 * SCALE,
            pnl=100 * SCALE,
            turnover=500 * SCALE,
            mark_price=100 * SCALE,
        )
        
        with pytest.raises(Exception):  # frozen dataclass
            snapshot.usd_balance = 9999 * SCALE


class TestNegativeBalances:
    """Test negative balance scenarios (infinite money model)."""

    def test_negative_usd_balance(self) -> None:
        """Test that USD can go negative."""
        inv = Inventory(initial_usd=0)
        
        # Buy more than we have
        buy_report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=100 * SCALE,
            filled_amount=100 * SCALE,
            timestamp=datetime.now(),
        )
        inv.process_fill(buy_report, Side.BUY, 1000000)
        
        assert inv._usd_balance == -10000 * SCALE  # -$10,000
        assert inv._token_balance == 100 * SCALE

    def test_negative_token_balance(self) -> None:
        """Test that tokens can go negative (short selling)."""
        inv = Inventory(initial_usd=10000 * SCALE, initial_tokens=0)
        
        # Sell tokens we don't have
        sell_report = ExecutionReport(
            report_id=uuid4(),
            order_id=uuid4(),
            status=ExecutionStatus.FILLED,
            filled_price=100 * SCALE,
            filled_amount=50 * SCALE,
            timestamp=datetime.now(),
        )
        inv.process_fill(sell_report, Side.SELL, 1000000)
        
        assert inv._token_balance == -50 * SCALE
        assert inv._usd_balance == 10000 * SCALE + 5000 * SCALE
