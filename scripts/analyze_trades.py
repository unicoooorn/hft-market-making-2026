#!/usr/bin/env python3
"""Analyze trades data and plot statistics."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow
import pyarrow.csv as csv


def analyze_trades(csv_path: Path, sample_size: int = 1_000_000) -> None:
    """Load trades CSV and plot statistics.

    Args:
        csv_path: Path to trades.csv file
        sample_size: Number of rows to sample for plotting
    """
    print(f"Loading {csv_path}...")

    table = csv.read_csv(
        csv_path,
        convert_options=csv.ConvertOptions(
            column_types={
                "": pyarrow.int64(),
                "local_timestamp": pyarrow.int64(),
                "side": pyarrow.string(),
                "price": pyarrow.float64(),
                "amount": pyarrow.float64(),
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
    sides = table.column("side").to_numpy()
    prices = table.column("price").to_numpy()
    amounts = table.column("amount").to_numpy()

    print(f"Loaded {len(timestamps):,} rows")

    # Calculate statistics
    buy_mask = sides == "buy"
    sell_mask = sides == "sell"

    buy_amounts = amounts[buy_mask]
    sell_amounts = amounts[sell_mask]
    buy_prices = prices[buy_mask]
    sell_prices = prices[sell_mask]

    # Trade volumes (amounts)
    all_amounts = amounts

    # Time between trades
    time_diffs = np.diff(timestamps) / 1_000_000  # seconds
    trade_rates = 1.0 / time_diffs[time_diffs > 0]  # trades per second

    # Convert timestamps to datetime
    timestamps_dt = [datetime.fromtimestamp(ts / 1_000_000) for ts in timestamps]

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.suptitle(f"Trades Analysis - {len(timestamps):,} samples", fontsize=16, fontweight="bold")

    # 1. Trade amounts over time
    ax1 = axes[0, 0]
    ax1.scatter(timestamps_dt[:5000], amounts[:5000], s=0.5, alpha=0.3, c="blue")
    ax1.set_title(f"Trade Amounts (first 5k trades)", fontweight="bold")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amount")
    ax1.grid(True, alpha=0.3)

    # 2. Trade amounts histogram
    ax2 = axes[0, 1]
    ax2.hist(all_amounts, bins=100, color="blue", alpha=0.7, edgecolor="black", log=True)
    ax2.set_title(f"Trade Amount Distribution (log scale)", fontweight="bold")
    ax2.set_xlabel("Amount")
    ax2.set_ylabel("Frequency (log)")
    ax2.grid(True, alpha=0.3)

    # 3. Price over time
    ax3 = axes[1, 0]
    ax3.plot(timestamps_dt[:5000], prices[:5000], "green", linewidth=0.5, alpha=0.7)
    ax3.set_title(f"Trade Prices (first 5k)", fontweight="bold")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Price")
    ax3.grid(True, alpha=0.3)

    # 4. Price histogram
    ax4 = axes[1, 1]
    ax4.hist(prices, bins=100, color="green", alpha=0.7, edgecolor="black")
    ax4.set_title(f"Trade Price Distribution", fontweight="bold")
    ax4.set_xlabel("Price")
    ax4.set_ylabel("Frequency")
    ax4.grid(True, alpha=0.3)

    # 5. Buy vs Sell amounts
    ax5 = axes[2, 0]
    ax5.hist([buy_amounts, sell_amounts], bins=100, color=["blue", "red"], alpha=0.5, label=["Buy", "Sell"], log=True)
    ax5.set_title(f"Buy vs Sell Amounts", fontweight="bold")
    ax5.set_xlabel("Amount")
    ax5.set_ylabel("Frequency (log)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Trade rate (trades per second)
    ax6 = axes[2, 1]
    if len(trade_rates) > 0:
        ax6.hist(trade_rates, bins=100, color="purple", alpha=0.7, edgecolor="black")
        ax6.set_title(f"Trade Rate Distribution", fontweight="bold")
        ax6.set_xlabel("Trades/Second")
        ax6.set_ylabel("Frequency")
    else:
        ax6.text(0.5, 0.5, "No data", ha="center", va="center")
    ax6.grid(True, alpha=0.3)

    # 7. Cumulative volume over time
    ax7 = axes[3, 0]
    cumulative_volume = np.cumsum(all_amounts)
    ax7.plot(timestamps_dt, cumulative_volume, "orange", linewidth=0.5)
    ax7.set_title(f"Cumulative Volume: {cumulative_volume[-1]:,.0f}", fontweight="bold")
    ax7.set_xlabel("Time")
    ax7.set_ylabel("Cumulative Amount")
    ax7.grid(True, alpha=0.3)

    # 8. Buy/Sell ratio over time (rolling)
    ax8 = axes[3, 1]
    window = 1000
    buy_counts = np.convolve(buy_mask.astype(int), np.ones(window), mode="valid")
    sell_counts = np.convolve(sell_mask.astype(int), np.ones(window), mode="valid")
    ratios = buy_counts / (sell_counts + 1e-6)
    x_axis = list(range(len(ratios)))
    ax8.plot(x_axis, ratios, "brown", linewidth=0.5)
    ax8.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax8.set_title(f"Buy/Sell Ratio (rolling {window})", fontweight="bold")
    ax8.set_xlabel("Trade Index")
    ax8.set_ylabel("Buy/Sell Ratio")
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 70)
    print("TRADES SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nOverall Statistics:")
    print(f"  Total Trades: {len(all_amounts):,}")
    print(f"  Buy Trades: {buy_mask.sum():,} ({100*buy_mask.mean():.1f}%)")
    print(f"  Sell Trades: {sell_mask.sum():,} ({100*sell_mask.mean():.1f}%)")

    print(f"\nTrade Volume (Amount) Statistics:")
    print(f"  Min: {all_amounts.min():,.0f}")
    print(f"  Max: {all_amounts.max():,.0f}")
    print(f"  Mean: {all_amounts.mean():,.0f}")
    print(f"  Median (p50): {np.median(all_amounts):,.0f}")
    print(f"  p25: {np.percentile(all_amounts, 25):,.0f}")
    print(f"  p75: {np.percentile(all_amounts, 75):,.0f}")
    print(f"  p90: {np.percentile(all_amounts, 90):,.0f}")
    print(f"  p99: {np.percentile(all_amounts, 99):,.0f}")
    print(f"  Total Volume: {all_amounts.sum():,.0f}")

    print(f"\nBuy Volume Statistics:")
    print(f"  Min: {buy_amounts.min():,.0f}")
    print(f"  Max: {buy_amounts.max():,.0f}")
    print(f"  Mean: {buy_amounts.mean():,.0f}")
    print(f"  Median (p50): {np.median(buy_amounts):,.0f}")
    print(f"  p25: {np.percentile(buy_amounts, 25):,.0f}")
    print(f"  p75: {np.percentile(buy_amounts, 75):,.0f}")
    print(f"  Total: {buy_amounts.sum():,.0f}")

    print(f"\nSell Volume Statistics:")
    print(f"  Min: {sell_amounts.min():,.0f}")
    print(f"  Max: {sell_amounts.max():,.0f}")
    print(f"  Mean: {sell_amounts.mean():,.0f}")
    print(f"  Median (p50): {np.median(sell_amounts):,.0f}")
    print(f"  p25: {np.percentile(sell_amounts, 25):,.0f}")
    print(f"  p75: {np.percentile(sell_amounts, 75):,.0f}")
    print(f"  Total: {sell_amounts.sum():,.0f}")

    print(f"\nPrice Statistics:")
    print(f"  Min: {prices.min():.8f}")
    print(f"  Max: {prices.max():.8f}")
    print(f"  Mean: {prices.mean():.8f}")
    print(f"  Median: {np.median(prices):.8f}")

    print(f"\nTrade Timing:")
    if len(time_diffs) > 0:
        print(f"  Mean time between trades: {np.mean(time_diffs):.3f}s")
        print(f"  Median time between trades: {np.median(time_diffs):.3f}s")
        print(f"  Mean trade rate: {np.mean(trade_rates):.1f} trades/sec")
        print(f"  Median trade rate: {np.median(trade_rates):.1f} trades/sec")

    print("=" * 70)


def main() -> None:
    """Run trades analysis."""
    base_dir = Path(__file__).parent.parent
    csv_path = base_dir / "data" / "cmf" / "trades.csv"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found!")
        return

    analyze_trades(csv_path, sample_size=1_000_000)


if __name__ == "__main__":
    main()
