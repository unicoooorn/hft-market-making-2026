from pathlib import Path

from backtester.data.provider import CSVDataProvider
from backtester.model import Side, OrderBookSnapshot, TradeEvent


DATA_DIR = Path(__file__).parent.parent / "data" / "cmf"

SCALE = 10**8


class TestCSVDataProvider:
    """Tests for streaming CSV data provider with Parquet."""

    def test_first_event_is_trade(self) -> None:
        """Test that first event from provider is a trade (earlier timestamp)."""
        provider = CSVDataProvider(
            order_book_path=DATA_DIR / "lob.parquet",
            trades_path=DATA_DIR / "trades.parquet",
        )

        first_event = provider.next()

        assert first_event is not None
        assert isinstance(first_event, TradeEvent)
        assert first_event.trade_seq == 0

    def test_parses_first_trade_correctly(self) -> None:
        """Test parsing the first trade event."""
        provider = CSVDataProvider(
            order_book_path=DATA_DIR / "lob.parquet",
            trades_path=DATA_DIR / "trades.parquet",
        )

        first_event = provider.next()
        assert isinstance(first_event, TradeEvent)

        assert first_event.symbol == "CMF"
        assert first_event.trade_seq == 0
        assert first_event.side == Side.SELL
        assert first_event.price > 0
        assert first_event.amount > 0

    def test_parses_first_order_book_correctly(self) -> None:
        """Test parsing the first order book snapshot (top of book)."""
        provider = CSVDataProvider(
            order_book_path=DATA_DIR / "lob.parquet",
            trades_path=DATA_DIR / "trades.parquet",
        )

        first_ob = None
        for _ in range(100):
            event = provider.next()
            if event is None:
                break
            if isinstance(event, OrderBookSnapshot):
                first_ob = event
                break

        assert first_ob is not None
        assert first_ob.symbol == "CMF"
        assert first_ob.snapshot_seq >= 0
        assert first_ob.bid_price > 0
        assert first_ob.ask_price > 0
        assert first_ob.bid_amount > 0
        assert first_ob.ask_amount > 0

    def test_events_are_ordered_by_timestamp(self) -> None:
        """Test that events are returned in timestamp order."""
        provider = CSVDataProvider(
            order_book_path=DATA_DIR / "lob.parquet",
            trades_path=DATA_DIR / "trades.parquet",
        )

        prev_ts = None
        for _ in range(100):
            event = provider.next()
            if event is None:
                break

            if prev_ts is not None:
                assert event.timestamp_us >= prev_ts
            prev_ts = event.timestamp_us

    def test_returns_none_when_exhausted(self) -> None:
        """Test that next() returns None when both files are exhausted."""
        provider = CSVDataProvider(
            order_book_path=DATA_DIR / "lob.parquet",
            trades_path=DATA_DIR / "trades.parquet",
        )

        for _ in range(10):
            event = provider.next()
            assert event is not None
