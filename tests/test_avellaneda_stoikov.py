"""Tests for Avellaneda-Stoikov market making strategy."""

from datetime import datetime
from uuid import uuid4

import pytest

from backtester.inventory import Inventory
from backtester.model import (
    OrderBookSnapshot,
    TradeEvent,
    Side,
)
from backtester.strategy.avellaneda_stoikov import (
    AvellanedaStoikovConfig,
    AvellanedaStoikovStrategy,
    QuoteCalculator,
    VolatilityEstimator,
)

SCALE = 10**8


class TestVolatilityEstimator:
    """Tests for volatility estimation."""

    def test_insufficient_data(self) -> None:
        """Test volatility returns None with insufficient data."""
        est = VolatilityEstimator(window_size=10)
        assert est.volatility() is None

        est.update(100.0, 1000000)
        assert est.volatility() is None

    def test_rolling_window(self) -> None:
        """Test that estimator respects window size."""
        est = VolatilityEstimator(window_size=3)

        for i in range(10):
            est.update(100.0 + i, i * 1000000)

        assert len(est._prices) == 3

    def test_volatility_calculation(self) -> None:
        """Test basic volatility calculation."""
        est = VolatilityEstimator(window_size=100)

        for i in range(50):
            price = 100.0 + (i % 5) * 0.1
            est.update(price, i * 1000000)

        vol = est.volatility()
        assert vol is not None
        assert vol > 0


class TestQuoteCalculator:
    """Tests for quote calculation math."""

    def test_reservation_price_no_inventory(self) -> None:
        """Test reservation price with zero inventory."""
        config = AvellanedaStoikovConfig(gamma=0.1, k=0.5)
        calc = QuoteCalculator(config)

        r = calc.reservation_price(
            mid_price=100.0,
            inventory=0,
            volatility=0.2,
            tau=1.0 / 365,
        )

        assert r == 100.0

    def test_reservation_price_long_inventory(self) -> None:
        """Test reservation price decreases with long position."""
        config = AvellanedaStoikovConfig(gamma=0.1, k=0.5)
        calc = QuoteCalculator(config)

        r = calc.reservation_price(
            mid_price=100.0,
            inventory=100 * SCALE,
            volatility=0.2,
            tau=1.0 / 365,
        )

        assert r < 100.0

    def test_reservation_price_short_inventory(self) -> None:
        """Test reservation price increases with short position."""
        config = AvellanedaStoikovConfig(gamma=0.1, k=0.5)
        calc = QuoteCalculator(config)

        r = calc.reservation_price(
            mid_price=100.0,
            inventory=-100 * SCALE,
            volatility=0.2,
            tau=1.0 / 365,
        )

        assert r > 100.0

    def test_optimal_spread_positive(self) -> None:
        """Test optimal spread is always positive."""
        config = AvellanedaStoikovConfig(gamma=0.1, k=0.5)
        calc = QuoteCalculator(config)

        delta = calc.optimal_spread(volatility=0.2, tau=1.0 / 365)
        assert delta > 0

    def test_optimal_spread_increases_with_volatility(self) -> None:
        """Test spread widens with higher volatility."""
        config = AvellanedaStoikovConfig(gamma=0.1, k=0.5)
        calc = QuoteCalculator(config)

        delta_low = calc.optimal_spread(volatility=0.1, tau=1.0 / 365)
        delta_high = calc.optimal_spread(volatility=0.5, tau=1.0 / 365)

        assert delta_high > delta_low

    def test_compute_quotes_spread(self) -> None:
        """Test that ask > bid in computed quotes."""
        config = AvellanedaStoikovConfig(gamma=0.1, k=0.5)
        calc = QuoteCalculator(config)

        bid, ask = calc.compute_quotes(
            mid_price=100.0,
            inventory=0,
            volatility=0.2,
            tau=1.0 / 365,
        )

        assert bid < ask
        assert abs((bid + ask) / 2 - 100.0) < 0.5

    def test_apply_microstructure_tick_rounding(self) -> None:
        """Test prices are rounded to tick size."""
        config = AvellanedaStoikovConfig(tick_size=0.01)
        calc = QuoteCalculator(config)

        result = calc.apply_microstructure(
            bid=100.123,
            ask=100.456,
            best_bid=100.0,
            best_ask=101.0,
        )

        assert result is not None
        bid, ask = result
        assert bid == round(bid, 2)
        assert ask == round(ask, 2)

    def test_apply_microstructure_min_spread(self) -> None:
        """Test minimum spread is enforced."""
        config = AvellanedaStoikovConfig(min_spread=0.05, tick_size=0.01)
        calc = QuoteCalculator(config)

        result = calc.apply_microstructure(
            bid=100.0,
            ask=100.01,
            best_bid=99.9,
            best_ask=100.1,
        )

        assert result is not None
        bid, ask = result
        assert ask - bid >= 0.05

    def test_apply_microstructure_no_crossing(self) -> None:
        """Test quotes never cross."""
        config = AvellanedaStoikovConfig(tick_size=0.01)
        calc = QuoteCalculator(config)

        result = calc.apply_microstructure(
            bid=100.0,
            ask=100.0,
            best_bid=99.9,
            best_ask=100.1,
        )

        if result is not None:
            bid, ask = result
            assert bid < ask

    def test_apply_microstructure_empty_book(self) -> None:
        """Test handling of empty book (zero prices)."""
        config = AvellanedaStoikovConfig(tick_size=0.01)
        calc = QuoteCalculator(config)

        result = calc.apply_microstructure(
            bid=100.0,
            ask=100.1,
            best_bid=0,
            best_ask=0,
        )

        assert result is not None


class TestAvellanedaStoikovStrategy:
    """Tests for full strategy integration."""

    def test_initialization(self) -> None:
        """Test strategy initializes correctly."""
        inventory = Inventory()
        config = AvellanedaStoikovConfig()
        strategy = AvellanedaStoikovStrategy(inventory, config)

        assert strategy.strategy_id == "as_mm"
        assert strategy.get_volatility() is None

    def test_empty_book_returns_no_orders(self) -> None:
        """Test strategy handles empty book gracefully."""
        inventory = Inventory()
        strategy = AvellanedaStoikovStrategy(inventory)

        snapshot = OrderBookSnapshot(
            snapshot_seq=0,
            timestamp_us=1000000,
            bid_price=0,
            bid_amount=0,
            ask_price=0,
            ask_amount=0,
        )

        intents = strategy.on_order_book_snapshot(snapshot)
        assert intents == []

    def test_valid_book_returns_orders(self) -> None:
        """Test strategy generates orders for valid book."""
        inventory = Inventory()
        strategy = AvellanedaStoikovStrategy(inventory)

        snapshot = OrderBookSnapshot(
            snapshot_seq=0,
            timestamp_us=1000000,
            bid_price=100 * SCALE,
            bid_amount=1000 * SCALE,
            ask_price=101 * SCALE,
            ask_amount=1000 * SCALE,
        )

        intents = strategy.on_order_book_snapshot(snapshot)
        assert len(intents) == 2

        buy_intent = intents[0]
        sell_intent = intents[1]

        assert buy_intent.side == "buy"
        assert sell_intent.side == "sell"
        assert buy_intent.limit_price is not None
        assert sell_intent.limit_price is not None
        assert buy_intent.limit_price < sell_intent.limit_price

    def test_inventory_affects_quotes(self) -> None:
        """Test that inventory position affects quote prices."""
        inventory_long = Inventory(initial_tokens=10000 * SCALE)
        inventory_short = Inventory(initial_tokens=-10000 * SCALE)

        config = AvellanedaStoikovConfig(
            gamma=1.0,
            k=0.5,
            horizon_seconds=3600,
        )
        strategy_long = AvellanedaStoikovStrategy(inventory_long, config)
        strategy_short = AvellanedaStoikovStrategy(inventory_short, config)

        for i in range(50):
            price_variation = (i % 10) * 0.01
            snapshot = OrderBookSnapshot(
                snapshot_seq=i,
                timestamp_us=i * 1000000,
                bid_price=int((100 + price_variation) * SCALE),
                bid_amount=1000 * SCALE,
                ask_price=int((101 + price_variation) * SCALE),
                ask_amount=1000 * SCALE,
            )
            strategy_long.on_order_book_snapshot(snapshot)
            strategy_short.on_order_book_snapshot(snapshot)

        r_long = strategy_long.get_reservation_price()
        r_short = strategy_short.get_reservation_price()

        assert r_long is not None
        assert r_short is not None
        assert r_long < r_short

    def test_trade_event_updates_volatility(self) -> None:
        """Test that trade events update volatility estimate."""
        inventory = Inventory()
        strategy = AvellanedaStoikovStrategy(inventory)

        assert strategy.get_volatility() is None

        for i in range(50):
            event = TradeEvent(
                trade_seq=i,
                timestamp_us=i * 1000000,
                side=Side.BUY,
                price=(100 + i % 3) * SCALE,
                amount=10 * SCALE,
            )
            strategy.on_trade_event(event)

        vol = strategy.get_volatility()
        assert vol is not None
        assert vol > 0

    def test_reservation_price_monitoring(self) -> None:
        """Test reservation price monitoring method."""
        inventory = Inventory()
        strategy = AvellanedaStoikovStrategy(inventory)

        assert strategy.get_reservation_price() is None

        snapshot = OrderBookSnapshot(
            snapshot_seq=0,
            timestamp_us=1000000,
            bid_price=100 * SCALE,
            bid_amount=1000 * SCALE,
            ask_price=101 * SCALE,
            ask_amount=1000 * SCALE,
        )

        strategy.on_order_book_snapshot(snapshot)
        r = strategy.get_reservation_price()

        assert r is not None
        assert 99 < r < 102

    def test_custom_config(self) -> None:
        """Test strategy with custom configuration."""
        inventory = Inventory()
        config = AvellanedaStoikovConfig(
            gamma=0.5,
            k=1.0,
            horizon_seconds=300,
            min_spread=0.001,
            max_spread=0.02,
            tick_size=0.001,
        )
        strategy = AvellanedaStoikovStrategy(inventory, config)

        snapshot = OrderBookSnapshot(
            snapshot_seq=0,
            timestamp_us=1000000,
            bid_price=100 * SCALE,
            bid_amount=1000 * SCALE,
            ask_price=101 * SCALE,
            ask_amount=1000 * SCALE,
        )

        intents = strategy.on_order_book_snapshot(snapshot)
        assert len(intents) == 2

    def test_order_amount_from_config(self) -> None:
        """Test order amount comes from config."""
        inventory = Inventory()
        config = AvellanedaStoikovConfig(order_amount=500 * SCALE)
        strategy = AvellanedaStoikovStrategy(inventory, config)

        snapshot = OrderBookSnapshot(
            snapshot_seq=0,
            timestamp_us=1000000,
            bid_price=100 * SCALE,
            bid_amount=1000 * SCALE,
            ask_price=101 * SCALE,
            ask_amount=1000 * SCALE,
        )

        intents = strategy.on_order_book_snapshot(snapshot)
        assert intents[0].amount == 500 * SCALE
        assert intents[1].amount == 500 * SCALE
