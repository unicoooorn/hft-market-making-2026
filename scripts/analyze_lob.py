#!/usr/bin/env python3
"""Analyze order book data and plot market statistics."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow
import pyarrow.csv as csv


def analyze_lob(csv_path: Path, sample_size: int = 100_000) -> None:
    """Load LOB CSV and plot market statistics.

    Args:
        csv_path: Path to lob.csv file
        sample_size: Number of rows to sample for plotting
    """
    print(f"Loading {csv_path}...")

    table = csv.read_csv(
        csv_path,
        convert_options=csv.ConvertOptions(
            column_types={
                "": pyarrow.int64(),
                "local_timestamp": pyarrow.int64(),
                "bids[0].price": pyarrow.float64(),
                "bids[0].amount": pyarrow.float64(),
                "asks[0].price": pyarrow.float64(),
                "asks[0].amount": pyarrow.float64(),
            }
        ),
    )

    print(f"Total rows: {table.num_rows:,}")

    # Sample sequentially if larger than requested
    if table.num_rows > sample_size:
        step = table.num_rows // sample_size
        indices = list(range(0, table.num_rows, step))[:sample_size]
        table = table.take(indices)
        print(f"Sampled {len(indices):,} rows (step={step})")

    # Extract columns
    timestamps = table.column("local_timestamp").to_numpy()
    bid_price = table.column("bids[0].price").to_numpy()
    ask_price = table.column("asks[0].price").to_numpy()
    bid_amount = table.column("bids[0].amount").to_numpy()
    ask_amount = table.column("asks[0].amount").to_numpy()

    print(f"Loaded {len(timestamps):,} rows")

    # Calculate statistics
    spread = ask_price - bid_price
    mid_price = (bid_price + ask_price) / 2
    spread_pct = (spread / mid_price) * 100

    # Convert timestamps to datetime for plotting
    timestamps_dt = [datetime.fromtimestamp(ts / 1_000_000) for ts in timestamps]

    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f"LOB Analysis - {len(timestamps):,} samples", fontsize=16, fontweight="bold")

    # 1. Best Bid and Ask prices
    ax1 = axes[0, 0]
    ax1.plot(timestamps_dt, bid_price, "b-", linewidth=0.5, alpha=0.7, label="Best Bid")
    ax1.plot(timestamps_dt, ask_price, "r-", linewidth=0.5, alpha=0.7, label="Best Ask")
    ax1.set_title("Best Bid/Ask Prices", fontweight="bold")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Spread (absolute)
    ax2 = axes[0, 1]
    ax2.plot(timestamps_dt, spread, "g-", linewidth=0.5, alpha=0.7)
    ax2.set_title(f"Spread (abs) - Mean: {spread.mean():.8f}, Median: {np.median(spread):.8f}", fontweight="bold")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Spread")
    ax2.grid(True, alpha=0.3)

    # 3. Spread percentage
    ax3 = axes[1, 0]
    ax3.plot(timestamps_dt, spread_pct, "purple", linewidth=0.5, alpha=0.7)
    ax3.set_title(f"Spread % - Mean: {spread_pct.mean():.6f}%, Median: {np.median(spread_pct):.6f}%", fontweight="bold")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Spread %")
    ax3.grid(True, alpha=0.3)

    # 4. Mid price
    ax4 = axes[1, 1]
    ax4.plot(timestamps_dt, mid_price, "orange", linewidth=0.5, alpha=0.7)
    ax4.set_title(f"Mid Price - Mean: {mid_price.mean():.6f}", fontweight="bold")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Mid Price")
    ax4.grid(True, alpha=0.3)

    # 5. Bid/Ask amounts
    ax5 = axes[2, 0]
    ax5.plot(timestamps_dt, bid_amount, "b-", linewidth=0.5, alpha=0.7, label="Bid Amount")
    ax5.plot(timestamps_dt, ask_amount, "r-", linewidth=0.5, alpha=0.7, label="Ask Amount")
    ax5.set_title(f"Top-of-Book Liquidity - Bid Mean: {bid_amount.mean():,.0f}, Ask Mean: {ask_amount.mean():,.0f}", fontweight="bold")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Amount")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Spread histogram
    ax6 = axes[2, 1]
    ax6.hist(spread, bins=100, color="green", alpha=0.7, edgecolor="black")
    ax6.set_title(f"Spread Distribution - Median: {np.median(spread):.8f}, Max: {spread.max():.8f}", fontweight="bold")
    ax6.set_xlabel("Spread")
    ax6.set_ylabel("Frequency")
    ax6.grid(True, alpha=0.3)

    # 7. Spread % histogram
    ax7 = axes[3, 0]
    ax7.hist(spread_pct, bins=100, color="purple", alpha=0.7, edgecolor="black")
    ax7.set_title(f"Spread % Distribution - Median: {np.median(spread_pct):.6f}%, Max: {spread_pct.max():.6f}%", fontweight="bold")
    ax7.set_xlabel("Spread %")
    ax7.set_ylabel("Frequency")
    ax7.grid(True, alpha=0.3)

    # 8. Price vs Spread scatter (sampled)
    sample_step = max(1, len(mid_price) // 5000)
    ax8 = axes[3, 1]
    ax8.scatter(mid_price[::sample_step], spread[::sample_step], s=0.5, alpha=0.3, c="blue")
    ax8.set_title("Mid Price vs Spread (sampled)", fontweight="bold")
    ax8.set_xlabel("Mid Price")
    ax8.set_ylabel("Spread")
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nPrice Statistics:")
    print(f"  Best Bid  - Min: {bid_price.min():.8f}, Max: {bid_price.max():.8f}, Mean: {bid_price.mean():.8f}")
    print(f"  Best Ask  - Min: {ask_price.min():.8f}, Max: {ask_price.max():.8f}, Mean: {ask_price.mean():.8f}")
    print(f"  Mid Price - Min: {mid_price.min():.8f}, Max: {mid_price.max():.8f}, Mean: {mid_price.mean():.8f}")
    print(f"\nSpread Statistics:")
    print(f"  Absolute - Min: {spread.min():.8f}, Max: {spread.max():.8f}, Mean: {spread.mean():.8f}, Std: {spread.std():.8f}")
    print(f"             Median: {np.median(spread):.8f}")
    print(f"  Percent  - Min: {spread_pct.min():.6f}%, Max: {spread_pct.max():.6f}%, Mean: {spread_pct.mean():.6f}%, Std: {spread_pct.std():.6f}%")
    print(f"             Median: {np.median(spread_pct):.6f}%")
    print(f"\nLiquidity Statistics:")
    print(f"  Bid Amount  - Min: {bid_amount.min():,.0f}, Max: {bid_amount.max():,.0f}, Mean: {bid_amount.mean():,.0f}, Median: {np.median(bid_amount):,.0f}")
    print(f"  Ask Amount  - Min: {ask_amount.min():,.0f}, Max: {ask_amount.max():,.0f}, Mean: {ask_amount.mean():,.0f}, Median: {np.median(ask_amount):,.0f}")
    print("=" * 70)


def main() -> None:
    """Run LOB analysis."""
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "cmf" / "lob.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return

    analyze_lob(csv_path, sample_size=100_000)


if __name__ == "__main__":
    main()
