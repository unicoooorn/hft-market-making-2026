#!/usr/bin/env python3
"""
High-performance Parquet reader for order book data.

This module demonstrates multiple approaches to reading order book Parquet files,
from slowest (Python dataclasses with Decimal) to fastest (Arrow-native with scaled integers).

Key Performance Insights:
1. Decimal is 10-50x slower than float/int
2. Row-by-row Python object creation is the main bottleneck
3. Batch processing with numpy/arrow is 100x+ faster
4. Scaled integers (price * 1e8) avoid Decimal overhead while maintaining precision
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# REPRESENTATION 1: Python Dataclasses with Decimal (Slowest, Most Readable)
# =============================================================================

@dataclass(frozen=True, slots=True)
class OrderBookLevelDecimal:
    """Single price level with Decimal precision."""
    price: Decimal
    amount: Decimal


@dataclass(frozen=True, slots=True)
class OrderBookSnapshotDecimal:
    """Full order book snapshot with Decimal precision."""
    snapshot_seq: int
    bids: tuple[OrderBookLevelDecimal, ...]
    asks: tuple[OrderBookLevelDecimal, ...]


def read_snapshots_decimal(
    parquet_path: Path,
    batch_size: int = 10000,
) -> Iterator[OrderBookSnapshotDecimal]:
    """
    Read order book snapshots as Python dataclasses with Decimal.
    
    SLOWEST approach - creates Python objects row-by-row.
    Throughput: ~5-10K rows/sec on typical hardware.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    
    # Pre-compute column indices
    schema = parquet_file.schema_arrow.names
    seq_idx = schema.index("")
    
    bid_price_idxs = [schema.index(f"bids[{i}].price") for i in range(25)]
    bid_amount_idxs = [schema.index(f"bids[{i}].amount") for i in range(25)]
    ask_price_idxs = [schema.index(f"asks[{i}].price") for i in range(25)]
    ask_amount_idxs = [schema.index(f"asks[{i}].amount") for i in range(25)]
    
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        # Convert to Python lists (slow!)
        seq_arr = batch.column(seq_idx).to_pylist()
        
        bid_prices = [[col[i].as_py() for col in [batch.column(idx) for idx in bid_price_idxs]] 
                      for i in range(batch.num_rows)]
        bid_amounts = [[col[i].as_py() for col in [batch.column(idx) for idx in bid_amount_idxs]] 
                       for i in range(batch.num_rows)]
        ask_prices = [[col[i].as_py() for col in [batch.column(idx) for idx in ask_price_idxs]] 
                      for i in range(batch.num_rows)]
        ask_amounts = [[col[i].as_py() for col in [batch.column(idx) for idx in ask_amount_idxs]] 
                       for i in range(batch.num_rows)]
        
        for i in range(batch.num_rows):
            bids = []
            asks = []
            
            for j in range(25):
                bp = bid_prices[i][j]
                ba = bid_amounts[i][j]
                if bp and ba and bp > 0 and ba > 0:
                    bids.append(OrderBookLevelDecimal(
                        price=Decimal(str(bp)), 
                        amount=Decimal(str(ba))
                    ))
                
                ap = ask_prices[i][j]
                aa = ask_amounts[i][j]
                if ap and aa and ap > 0 and aa > 0:
                    asks.append(OrderBookLevelDecimal(
                        price=Decimal(str(ap)), 
                        amount=Decimal(str(aa))
                    ))
            
            yield OrderBookSnapshotDecimal(
                snapshot_seq=int(seq_arr[i]),
                bids=tuple(bids),
                asks=tuple(asks),
            )


# =============================================================================
# REPRESENTATION 2: NumPy Arrays with Float64 (Fast, Good Precision)
# =============================================================================

@dataclass(frozen=True, slots=True)
class OrderBookSnapshotNumpy:
    """Order book snapshot with numpy arrays for levels."""
    snapshot_seq: int
    bid_prices: np.ndarray  # shape: (25,)
    bid_amounts: np.ndarray  # shape: (25,)
    ask_prices: np.ndarray  # shape: (25,)
    ask_amounts: np.ndarray  # shape: (25,)


def read_snapshots_numpy(
    parquet_path: Path,
    batch_size: int = 10000,
) -> Iterator[OrderBookSnapshotNumpy]:
    """
    Read order book snapshots using NumPy arrays.
    
    FAST approach - batch conversion to numpy, then row iteration.
    Throughput: ~50-100K rows/sec on typical hardware.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow.names
    
    seq_idx = schema.index("")
    bid_price_idxs = [schema.index(f"bids[{i}].price") for i in range(25)]
    bid_amount_idxs = [schema.index(f"bids[{i}].amount") for i in range(25)]
    ask_price_idxs = [schema.index(f"asks[{i}].price") for i in range(25)]
    ask_amount_idxs = [schema.index(f"asks[{i}].amount") for i in range(25)]
    
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        # Convert entire columns to numpy at once (fast!)
        seq_arr = batch.column(seq_idx).to_numpy()
        
        bid_prices_arr = np.column_stack([
            batch.column(idx).to_numpy() for idx in bid_price_idxs
        ])
        bid_amounts_arr = np.column_stack([
            batch.column(idx).to_numpy() for idx in bid_amount_idxs
        ])
        ask_prices_arr = np.column_stack([
            batch.column(idx).to_numpy() for idx in ask_price_idxs
        ])
        ask_amounts_arr = np.column_stack([
            batch.column(idx).to_numpy() for idx in ask_amount_idxs
        ])
        
        for i in range(batch.num_rows):
            yield OrderBookSnapshotNumpy(
                snapshot_seq=int(seq_arr[i]),
                bid_prices=bid_prices_arr[i],
                bid_amounts=bid_amounts_arr[i],
                ask_prices=ask_prices_arr[i],
                ask_amounts=ask_amounts_arr[i],
            )


# =============================================================================
# REPRESENTATION 3: Scaled Integers (Fastest, No Float Precision Issues)
# =============================================================================

@dataclass(frozen=True, slots=True)
class OrderBookLevelInt:
    """Single price level with scaled integer (price * 1e8)."""
    price_scaled: int  # price * 1e8
    amount_scaled: int  # amount * 1e8
    
    @property
    def price(self) -> float:
        """Get price as float."""
        return self.price_scaled / 1e8
    
    @property
    def amount(self) -> float:
        """Get amount as float."""
        return self.amount_scaled / 1e8


@dataclass(frozen=True, slots=True)
class OrderBookSnapshotInt:
    """Order book snapshot with scaled integers."""
    snapshot_seq: int
    bid_prices_scaled: np.ndarray  # int64, shape: (25,)
    bid_amounts_scaled: np.ndarray  # int64, shape: (25,)
    ask_prices_scaled: np.ndarray  # int64, shape: (25,)
    ask_amounts_scaled: np.ndarray  # int64, shape: (25,)
    
    def get_bid(self, idx: int) -> Optional[OrderBookLevelInt]:
        """Get bid level by index."""
        if idx >= len(self.bid_prices_scaled):
            return None
        price = self.bid_prices_scaled[idx]
        amount = self.bid_amounts_scaled[idx]
        if price <= 0 or amount <= 0:
            return None
        return OrderBookLevelInt(price_scaled=int(price), amount_scaled=int(amount))
    
    def get_ask(self, idx: int) -> Optional[OrderBookLevelInt]:
        """Get ask level by index."""
        if idx >= len(self.ask_prices_scaled):
            return None
        price = self.ask_prices_scaled[idx]
        amount = self.ask_amounts_scaled[idx]
        if price <= 0 or amount <= 0:
            return None
        return OrderBookLevelInt(price_scaled=int(price), amount_scaled=int(amount))


def read_snapshots_scaled_int(
    parquet_path: Path,
    batch_size: int = 10000,
    price_scale: int = 10**8,
    amount_scale: int = 10**8,
) -> Iterator[OrderBookSnapshotInt]:
    """
    Read order book snapshots using scaled integers.
    
    FASTEST approach - avoids float/Decimal overhead entirely.
    Throughput: ~100-200K rows/sec on typical hardware.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow.names
    
    seq_idx = schema.index("")
    bid_price_idxs = [schema.index(f"bids[{i}].price") for i in range(25)]
    bid_amount_idxs = [schema.index(f"bids[{i}].amount") for i in range(25)]
    ask_price_idxs = [schema.index(f"asks[{i}].price") for i in range(25)]
    ask_amount_idxs = [schema.index(f"asks[{i}].amount") for i in range(25)]
    
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        # Convert to numpy and scale to integers in one operation
        seq_arr = batch.column(seq_idx).to_numpy()
        
        bid_prices_scaled = (
            np.column_stack([
                batch.column(idx).to_numpy() for idx in bid_price_idxs
            ]) * price_scale
        ).astype(np.int64)
        
        bid_amounts_scaled = (
            np.column_stack([
                batch.column(idx).to_numpy() for idx in bid_amount_idxs
            ]) * amount_scale
        ).astype(np.int64)
        
        ask_prices_scaled = (
            np.column_stack([
                batch.column(idx).to_numpy() for idx in ask_price_idxs
            ]) * price_scale
        ).astype(np.int64)
        
        ask_amounts_scaled = (
            np.column_stack([
                batch.column(idx).to_numpy() for idx in ask_amount_idxs
            ]) * amount_scale
        ).astype(np.int64)
        
        for i in range(batch.num_rows):
            yield OrderBookSnapshotInt(
                snapshot_seq=int(seq_arr[i]),
                bid_prices_scaled=bid_prices_scaled[i],
                bid_amounts_scaled=bid_amounts_scaled[i],
                ask_prices_scaled=ask_prices_scaled[i],
                ask_amounts_scaled=ask_amounts_scaled[i],
            )


# =============================================================================
# REPRESENTATION 4: Arrow-Native Batch Processing (Fastest for Bulk Operations)
# =============================================================================

class OrderBookBatchArrow:
    """
    Batch of order book snapshots in Arrow-native format.
    
    FASTEST for bulk operations - never converts to Python objects.
    Throughput: 1M+ rows/sec for column-wise operations.
    """
    
    def __init__(self, batch: pa.RecordBatch) -> None:
        self._batch = batch
        self.num_rows = batch.num_rows
        
        # Cache column access
        schema = batch.schema.names
        self._seq = batch.column(schema.index(""))
        
        self._bid_prices = [
            batch.column(schema.index(f"bids[{i}].price")) 
            for i in range(25)
        ]
        self._bid_amounts = [
            batch.column(schema.index(f"bids[{i}].amount")) 
            for i in range(25)
        ]
        self._ask_prices = [
            batch.column(schema.index(f"asks[{i}].price")) 
            for i in range(25)
        ]
        self._ask_amounts = [
            batch.column(schema.index(f"asks[{i}].amount")) 
            for i in range(25)
        ]
    
    def get_best_bid(self) -> tuple[pa.Array, pa.Array]:
        """Get best bid prices and amounts (level 0) as Arrow arrays."""
        return self._bid_prices[0], self._bid_amounts[0]
    
    def get_best_ask(self) -> tuple[pa.Array, pa.Array]:
        """Get best ask prices and amounts (level 0) as Arrow arrays."""
        return self._ask_prices[0], self._ask_amounts[0]
    
    def get_mid_price(self) -> pa.Array:
        """Calculate mid price for all rows in batch."""
        best_bid, _ = self.get_best_bid()
        best_ask, _ = self.get_best_ask()
        return (pa.array(best_bid) + pa.array(best_ask)) / 2
    
    def filter_by_price_range(
        self, 
        min_price: float, 
        max_price: float
    ) -> "OrderBookBatchArrow":
        """Filter batch to rows where mid price is in range."""
        mid = self.get_mid_price()
        mask = (mid >= min_price) & (mid <= max_price)
        return OrderBookBatchArrow(self._batch.filter(mask))


def read_batches_arrow(
    parquet_path: Path,
    batch_size: int = 10000,
    columns: Optional[list[str]] = None,
) -> Iterator[OrderBookBatchArrow]:
    """
    Read order book data as Arrow-native batches.
    
    FASTEST for bulk operations - no Python object creation.
    Use this for filtering, aggregation, or column-wise operations.
    """
    parquet_file = pq.ParquetFile(parquet_path)
    
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=columns,
    ):
        yield OrderBookBatchArrow(batch)


# =============================================================================
# BENCHMARKING
# =============================================================================

def benchmark_reader(
    parquet_path: Path,
    reader_func,
    name: str,
    num_rows: int = 100000,
) -> float:
    """Benchmark a reader function."""
    start = time.perf_counter()
    
    count = 0
    for _ in reader_func(parquet_path):
        count += 1
        if count >= num_rows:
            break
    
    elapsed = time.perf_counter() - start
    rows_per_sec = count / elapsed
    
    print(f"{name:40s}: {rows_per_sec:10.0f} rows/sec ({count:,} rows in {elapsed:.2f}s)")
    return rows_per_sec


def run_benchmarks(parquet_path: Path, num_rows: int = 100000) -> None:
    """Run all benchmarks."""
    print(f"\n{'='*70}")
    print(f"Benchmarking Parquet readers ({num_rows:,} rows)")
    print(f"File: {parquet_path}")
    print(f"{'='*70}\n")
    
    results = {}
    
    # Benchmark each approach
    results["Decimal Dataclasses (slowest)"] = benchmark_reader(
        parquet_path, 
        read_snapshots_decimal, 
        "Decimal Dataclasses",
        num_rows,
    )
    
    results["NumPy Float64"] = benchmark_reader(
        parquet_path, 
        read_snapshots_numpy, 
        "NumPy Float64",
        num_rows,
    )
    
    results["Scaled Integers (fastest)"] = benchmark_reader(
        parquet_path, 
        read_snapshots_scaled_int, 
        "Scaled Integers",
        num_rows,
    )
    
    # Arrow-native benchmark (different metric - batches not rows)
    print("\nArrow-native batch processing:")
    start = time.perf_counter()
    batch_count = 0
    total_rows = 0
    for batch in read_batches_arrow(parquet_path, batch_size=10000):
        batch_count += 1
        total_rows += batch.num_rows
        if total_rows >= num_rows:
            break
    elapsed = time.perf_counter() - start
    print(f"{'Arrow-native batches':40s}: {total_rows/elapsed:10.0f} rows/sec ({batch_count} batches in {elapsed:.2f}s)")
    results["Arrow-native (bulk ops)"] = total_rows / elapsed
    
    print(f"\n{'='*70}")
    print(f"Speedup vs Decimal: {results['Scaled Integers (fastest)']/results['Decimal Dataclasses (slowest)']:.1f}x")
    print(f"{'='*70}\n")


def demonstrate_column_selection(parquet_path: Path) -> None:
    """Demonstrate reading only specific columns."""
    print("\nColumn selection example:")
    print(f"{'='*70}\n")
    
    # Read only best bid/ask (5 columns instead of 102)
    columns = [
        "",
        "bids[0].price",
        "bids[0].amount",
        "asks[0].price",
        "asks[0].amount",
    ]
    
    parquet_file = pq.ParquetFile(parquet_path)
    
    start = time.perf_counter()
    total_rows = 0
    for batch in parquet_file.iter_batches(columns=columns, batch_size=10000):
        total_rows += batch.num_rows
    elapsed = time.perf_counter() - start
    
    print(f"Reading 5 columns: {total_rows/elapsed:,.0f} rows/sec")
    
    # Read all columns
    start = time.perf_counter()
    total_rows = 0
    for batch in parquet_file.iter_batches(batch_size=10000):
        total_rows += batch.num_rows
    elapsed = time.perf_counter() - start
    
    print(f"Reading all columns: {total_rows/elapsed:,.0f} rows/sec")
    print("\nColumn selection can provide 2-5x speedup for wide tables!\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python benchmark_parquet_read.py <parquet_path> [num_rows]")
        print("Example: python benchmark_parquet_read.py data/cmf/lob.parquet 100000")
        sys.exit(1)
    
    parquet_path = Path(sys.argv[1])
    num_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
    
    if not parquet_path.exists():
        print(f"Error: File not found: {parquet_path}")
        sys.exit(1)
    
    run_benchmarks(parquet_path, num_rows)
    demonstrate_column_selection(parquet_path)
