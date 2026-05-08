from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from backtester.core.interfaces import Strategy
from backtester.inventory import Inventory
from backtester.model import (
    OrderBookSnapshot,
    OrderIntent,
    PriceScaled,
    AmountScaled,
    TimestampUs,
    TradeEvent,
    Side,
)

SCALE = 10**8


@dataclass(frozen=True, slots=True)
class AvellanedaStoikovConfig:
    """Configuration for Avellaneda-Stoikov market making strategy."""
    gamma: float = 0.1
    k: float = 0.5
    volatility_window: int = 100
    horizon_seconds: float = 60.0
    min_spread: float = 0.0001
    max_spread: float = 0.01
    tick_size: float = 0.0001
    order_amount: AmountScaled = 1000 * SCALE
    cancellation_threshold: float = 0.0002


@dataclass
class Quote:
    """Market making quote with bid and ask prices."""
    bid: PriceScaled
    ask: PriceScaled


class VolatilityEstimator:
    """Rolling volatility estimator using standard deviation of log returns."""

    def __init__(self, window_size: int) -> None:
        self._window_size = window_size
        self._prices: deque[float] = deque(maxlen=window_size)
        self._timestamps: deque[TimestampUs] = deque(maxlen=window_size)

    def update(self, mid_price: float, timestamp_us: TimestampUs) -> None:
        """Update estimator with new mid price."""
        self._prices.append(mid_price)
        self._timestamps.append(timestamp_us)

    def volatility(self) -> float | None:
        """
        Estimate annualized volatility from rolling window.

        Returns standard deviation of log returns, annualized.
        Returns None if insufficient data.
        """
        if len(self._prices) < 2:
            return None

        prices = list(self._prices)
        timestamps = list(self._timestamps)

        log_returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0 and prices[i] > 0:
                log_ret = math.log(prices[i] / prices[i - 1])
                log_returns.append(log_ret)

        if len(log_returns) < 2:
            return None

        mean_ret = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_ret) ** 2 for r in log_returns) / (len(log_returns) - 1)
        std_ret = math.sqrt(variance)

        if len(timestamps) >= 2 and timestamps[-1] > timestamps[0]:
            time_span_seconds = (timestamps[-1] - timestamps[0]) / 1_000_000
            if time_span_seconds > 0:
                observations_per_year = (365 * 24 * 3600) / time_span_seconds
                std_ret *= math.sqrt(observations_per_year)

        return std_ret


class QuoteCalculator:
    """Pure math implementation of Avellaneda-Stoikov quote calculation."""

    def __init__(self, config: AvellanedaStoikovConfig) -> None:
        self._config = config

    def reservation_price(
        self,
        mid_price: float,
        inventory: int,
        volatility: float,
        tau: float,
    ) -> float:
        """
        Calculate reservation price using AS formula.

        r = S - q * gamma * sigma^2 * tau

        Args:
            mid_price: Current mid price (S)
            inventory: Current position (q), scaled by 1e8
            volatility: Annualized volatility (sigma)
            tau: Time horizon in years

        Returns:
            Reservation price
        """
        inventory_scaled = inventory / SCALE
        risk_adjustment = inventory_scaled * self._config.gamma * (volatility ** 2) * tau
        return mid_price - risk_adjustment

    def optimal_spread(self, volatility: float, tau: float) -> float:
        """
        Calculate optimal half-spread using AS formula.

        delta = (1/gamma) * ln(1 + gamma/k) + 0.5 * gamma * sigma^2 * tau

        Args:
            volatility: Annualized volatility (sigma)
            tau: Time horizon in years

        Returns:
            Optimal half-spread
        """
        gamma = self._config.gamma
        k = self._config.k

        if k <= 0:
            k = 0.001

        term1 = (1 / gamma) * math.log(1 + gamma / k)
        term2 = 0.5 * gamma * (volatility ** 2) * tau

        return term1 + term2

    def compute_quotes(
        self,
        mid_price: float,
        inventory: int,
        volatility: float,
        tau: float,
    ) -> tuple[float, float]:
        """
        Compute bid and ask quotes.

        Args:
            mid_price: Current mid price
            inventory: Current position (scaled)
            volatility: Annualized volatility
            tau: Time horizon in years

        Returns:
            Tuple of (bid_price, ask_price)
        """
        r = self.reservation_price(mid_price, inventory, volatility, tau)
        delta = self.optimal_spread(volatility, tau)

        bid = r - delta
        ask = r + delta

        return bid, ask

    def apply_microstructure(
        self,
        bid: float,
        ask: float,
        best_bid: float,
        best_ask: float,
    ) -> tuple[float, float] | None:
        """
        Apply microstructure constraints to quotes.

        - Round to tick size
        - Enforce min/max spread
        - Prevent crossed quotes
        - Ensure quotes are within book

        Returns None if quotes cannot be validly constructed.
        """
        tick = self._config.tick_size
        min_spread = self._config.min_spread
        max_spread = self._config.max_spread

        bid_rounded = math.floor(bid / tick) * tick
        ask_rounded = math.ceil(ask / tick) * tick

        spread = ask_rounded - bid_rounded

        if spread < min_spread:
            adjustment = (min_spread - spread) / 2
            bid_rounded -= adjustment
            ask_rounded += adjustment
            bid_rounded = math.floor(bid_rounded / tick) * tick
            ask_rounded = math.ceil(ask_rounded / tick) * tick

        if spread > max_spread:
            mid = (bid_rounded + ask_rounded) / 2
            half_spread = max_spread / 2
            bid_rounded = mid - half_spread
            ask_rounded = mid + half_spread
            bid_rounded = math.floor(bid_rounded / tick) * tick
            ask_rounded = math.ceil(ask_rounded / tick) * tick

        if bid_rounded >= ask_rounded:
            return None

        if best_bid > 0 and bid_rounded >= best_bid:
            bid_rounded = best_bid - tick
            if bid_rounded <= 0:
                return None

        if best_ask > 0 and ask_rounded <= best_ask:
            ask_rounded = best_ask + tick

        if bid_rounded >= ask_rounded:
            return None

        return bid_rounded, ask_rounded


class AvellanedaStoikovStrategy(Strategy):
    """
    Avellaneda-Stoikov market making strategy.

    Uses inventory-aware pricing to manage risk while providing liquidity.
    """

    def __init__(
        self,
        inventory: Inventory,
        config: AvellanedaStoikovConfig | None = None,
        strategy_id: str = "as_mm",
    ) -> None:
        self.strategy_id = strategy_id
        self._inventory = inventory
        self._config = config or AvellanedaStoikovConfig()
        self._quote_calc = QuoteCalculator(self._config)
        self._vol_estimator = VolatilityEstimator(self._config.volatility_window)
        self._last_mid_price: float | None = None
        self._last_timestamp: TimestampUs | None = None
        self._orders_placed: bool = False
        self._placement_mid_price: float | None = None

    def on_order_book_snapshot(
        self,
        snapshot: OrderBookSnapshot,
    ) -> list[OrderIntent]:
        """Handle order book snapshot and return order intents."""
        if snapshot.bid_price <= 0 or snapshot.ask_price <= 0:
            return []

        best_bid = snapshot.bid_price / SCALE
        best_ask = snapshot.ask_price / SCALE
        mid_price = snapshot.mid_price

        if mid_price is None:
            return []

        self._vol_estimator.update(mid_price, snapshot.timestamp_us)
        self._last_mid_price = mid_price
        self._last_timestamp = snapshot.timestamp_us

        volatility = self._vol_estimator.volatility()
        if volatility is None:
            volatility = 0.1

        tau = self._config.horizon_seconds / (365 * 24 * 3600)
        inventory = self._inventory.token_balance

        bid_float, ask_float = self._quote_calc.compute_quotes(
            mid_price=mid_price,
            inventory=inventory,
            volatility=volatility,
            tau=tau,
        )

        micro = self._quote_calc.apply_microstructure(
            bid=bid_float,
            ask=ask_float,
            best_bid=best_bid,
            best_ask=best_ask,
        )

        if micro is None:
            return []

        bid_scaled, ask_scaled = micro
        bid_scaled_int = int(bid_scaled * SCALE)
        ask_scaled_int = int(ask_scaled * SCALE)

        now = datetime.now()
        intents: list[OrderIntent] = []

        intents.append(OrderIntent(
            order_id=uuid4(),
            symbol=snapshot.symbol,
            side="buy",
            order_type="limit",
            amount=self._config.order_amount,
            limit_price=bid_scaled_int,
            created_ts=now,
            cancel_and_replace=True,
        ))

        intents.append(OrderIntent(
            order_id=uuid4(),
            symbol=snapshot.symbol,
            side="sell",
            order_type="limit",
            amount=self._config.order_amount,
            limit_price=ask_scaled_int,
            created_ts=now,
            cancel_and_replace=True,
        ))

        self._orders_placed = True
        self._placement_mid_price = mid_price

        return intents

    def on_trade_event(
        self,
        event: TradeEvent,
    ) -> list[OrderIntent]:
        """Handle trade event - update volatility and check for cancellation."""
        if event.price > 0:
            price_float = event.price / SCALE
            self._vol_estimator.update(price_float, event.timestamp_us)
            self._last_mid_price = price_float
            self._last_timestamp = event.timestamp_us

            if self._should_cancel(price_float):
                return [self._create_cancel_intent()]

        return []

    def _should_cancel(self, current_mid_price: float) -> bool:
        """Check if orders should be cancelled based on midprice drift."""
        if not self._orders_placed or self._placement_mid_price is None:
            return False

        drift = abs(current_mid_price - self._placement_mid_price)
        return drift > self._config.cancellation_threshold

    def _create_cancel_intent(self) -> OrderIntent:
        """Create a cancel intent for both bid and ask."""
        self._orders_placed = False
        self._placement_mid_price = None

        return OrderIntent(
            order_id=uuid4(),
            symbol="CMF",
            side="cancel",
            order_type="cancel",
            amount=0,
            limit_price=None,
            created_ts=datetime.now(),
            cancel_and_replace=False,
        )

    def get_reservation_price(self) -> float | None:
        """Get current reservation price for monitoring."""
        if self._last_mid_price is None:
            return None

        volatility = self._vol_estimator.volatility()
        if volatility is None:
            return self._last_mid_price

        tau = self._config.horizon_seconds / (365 * 24 * 3600)
        inventory = self._inventory.token_balance

        return self._quote_calc.reservation_price(
            mid_price=self._last_mid_price,
            inventory=inventory,
            volatility=volatility,
            tau=tau,
        )

    def get_volatility(self) -> float | None:
        """Get current volatility estimate for monitoring."""
        return self._vol_estimator.volatility()
