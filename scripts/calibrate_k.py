#!/usr/bin/env python3
"""Calibrate k parameter for Avellaneda-Stoikov model.

Measures fill probability decay as function of distance from mid price.
Uses formula: log(λ) = log(A) - k*δ
where λ = resting_time / fills at distance δ
"""

from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


def load_data(lob_path: Path, trades_path: Path, sample_rows: int = 1_000_000):
    """Load LOB and trades data."""
    print(f"Loading LOB data...")
    lob_table = pq.read_table(lob_path)
    if lob_table.num_rows > sample_rows:
        lob_table = lob_table.slice(0, sample_rows)
    
    print(f"Loading trades data...")
    trades_table = pq.read_table(trades_path)
    if trades_table.num_rows > sample_rows * 20:
        trades_table = trades_table.slice(0, sample_rows * 20)
    
    return lob_table, trades_table


def extract_columns(lob_table, trades_table):
    """Extract necessary columns from tables."""
    SCALE = 1e8
    
    # LOB columns (scaled int64, convert to float)
    lob_ts = lob_table.column("local_timestamp").to_numpy()
    lob_bid = lob_table.column("bids[0].price").to_numpy() / SCALE
    lob_ask = lob_table.column("asks[0].price").to_numpy() / SCALE
    lob_bid_amt = lob_table.column("bids[0].amount").to_numpy() / SCALE
    lob_ask_amt = lob_table.column("asks[0].amount").to_numpy() / SCALE
    
    # Trades columns (scaled int64, convert to float)
    trades_ts = trades_table.column("local_timestamp").to_numpy()
    trades_price = trades_table.column("price").to_numpy() / SCALE
    trades_amount = trades_table.column("amount").to_numpy() / SCALE
    trades_side = trades_table.column("side").to_numpy()
    
    return {
        'lob_ts': lob_ts,
        'lob_bid': lob_bid,
        'lob_ask': lob_ask,
        'lob_bid_amt': lob_bid_amt,
        'lob_ask_amt': lob_ask_amt,
        'trades_ts': trades_ts,
        'trades_price': trades_price,
        'trades_amount': trades_amount,
        'trades_side': trades_side,
    }


def calculate_tick_size(bid_prices, ask_prices):
    """Calculate tick size as the most common non-zero price change."""
    bid_diffs = np.diff(bid_prices)
    ask_diffs = np.diff(ask_prices)
    all_diffs = np.concatenate([bid_diffs, ask_diffs])
    nonzero_diffs = np.abs(all_diffs[all_diffs != 0])
    
    # Round to 10 decimal places to handle floating point artifacts
    rounded = np.round(nonzero_diffs, 10)
    
    # Use mode (most common) instead of min to avoid outliers
    unique_ticks, counts = np.unique(rounded, return_counts=True)
    mode_idx = np.argmax(counts)
    tick = unique_ticks[mode_idx]
    
    print(f"Tick sizes found: {unique_ticks[:5]}")
    print(f"Calculated tick size (mode): {tick:.10f}")
    return tick


def simulate_order_placement(data, tick_size, max_ticks: int = 20):
    """Simulate placing limit orders at various distances from mid price."""
    lob_ts = data['lob_ts']
    lob_bid = data['lob_bid']
    lob_ask = data['lob_ask']
    trades_ts = data['trades_ts']
    trades_price = data['trades_price']
    trades_side = data['trades_side']
    
    n_lob = len(lob_ts)
    n_trades = len(trades_ts)
    
    resting_time = {i: 0.0 for i in range(1, max_ticks + 1)}
    fills = {i: 0 for i in range(1, max_ticks + 1)}
    
    print(f"Simulating order placement across {n_lob:,} LOB snapshots...")
    print(f"Processing {n_trades:,} trades...")
    print(f"LOB time range: {lob_ts[0]} - {lob_ts[-1]}")
    print(f"Trades time range: {trades_ts[0]} - {trades_ts[-1]}")
    
    debug_shown = False
    
    for i in range(n_lob - 1):
        mid_price = (lob_bid[i] + lob_ask[i]) / 2
        best_bid = lob_bid[i]
        best_ask = lob_ask[i]
        
        rest_duration = lob_ts[i + 1] - lob_ts[i]
        if rest_duration <= 0:
            continue
        
        current_ts = lob_ts[i]
        next_ts = lob_ts[i + 1]
        
        # Find trades in this window using binary search
        start_idx = np.searchsorted(trades_ts, current_ts, side='left')
        end_idx = np.searchsorted(trades_ts, next_ts, side='left')
        
        if start_idx >= end_idx:
            # No trades in this window
            for tick_distance in range(1, max_ticks + 1):
                resting_time[tick_distance] += rest_duration
            continue
        
        window_trades_side = trades_side[start_idx:end_idx]
        window_trades_price = trades_price[start_idx:end_idx]
        
        # Debug: show first window with trades
        if not debug_shown:
            print(f"\nFirst window with trades (LOB index {i}):")
            print(f"  Best bid: {best_bid:.8f}, Best ask: {best_ask:.8f}")
            print(f"  Window: {end_idx - start_idx} trades")
            debug_shown = True
        
        # Check each trade in window
        for j in range(len(window_trades_side)):
            trade_price = window_trades_price[j]
            trade_side = window_trades_side[j]
            
            for tick_distance in range(1, max_ticks + 1):
                # Bid order: at best bid (tick_distance=1 means at touch)
                bid_order_price = best_bid - (tick_distance - 1) * tick_size
                
                # Ask order: at best ask
                ask_order_price = best_ask + (tick_distance - 1) * tick_size
                
                # Bid fill: sell trade hits our bid
                if trade_side == 'sell' and trade_price <= bid_order_price:
                    fills[tick_distance] += 1
                
                # Ask fill: buy trade hits our ask
                if trade_side == 'buy' and trade_price >= ask_order_price:
                    fills[tick_distance] += 1
        
        # Accumulate resting time
        for tick_distance in range(1, max_ticks + 1):
            resting_time[tick_distance] += rest_duration
        
        if i % 100000 == 0 and i > 0:
            total_fills = sum(fills.values())
            print(f"  Progress: {i}/{n_lob} ({100*i/n_lob:.1f}%), fills: {total_fills}")
    
    total_fills = sum(fills.values())
    print(f"Total fills detected: {total_fills}")
    
    return resting_time, fills


def calculate_lambda(resting_time, fills):
    """Calculate λ(δ) = resting_time / fills for each distance."""
    lambdas = {}
    for tick_dist in resting_time.keys():
        if fills[tick_dist] > 0:
            # Convert resting time from microseconds to seconds
            resting_seconds = resting_time[tick_dist] / 1_000_000
            lambdas[tick_dist] = resting_seconds / fills[tick_dist]
        else:
            lambdas[tick_dist] = None
    return lambdas


def fit_k_parameter(lambdas):
    """Fit log(λ) = log(A) - k*δ using linear regression.
    
    λ = time_per_fill (higher = less likely to fill)
    In AS model: fill_intensity = A * exp(-k*δ)
    So: time_per_fill = 1/intensity = (1/A) * exp(k*δ)
    log(λ) = -log(A) + k*δ
    
    Returns:
        k: decay parameter (positive = fills decrease with distance)
        A: amplitude parameter (fill intensity at δ=0)
        r_squared: goodness of fit
    """
    # Filter out None values
    distances = []
    log_lambdas = []
    
    for tick_dist, lam in lambdas.items():
        if lam is not None and lam > 0:
            distances.append(tick_dist)
            log_lambdas.append(np.log(lam))
    
    if len(distances) < 2:
        return None, None, None
    
    distances = np.array(distances)
    log_lambdas = np.array(log_lambdas)
    
    # Linear regression: log(λ) = intercept + slope * δ
    # where slope = k (positive), intercept = -log(A)
    slope, intercept = np.polyfit(distances, log_lambdas, 1)
    
    # Calculate R-squared
    predicted = intercept + slope * distances
    ss_res = np.sum((log_lambdas - predicted) ** 2)
    ss_tot = np.sum((log_lambdas - np.mean(log_lambdas)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    k = slope  # Positive k means fills decrease with distance
    A = np.exp(-intercept)  # A = exp(-intercept) because intercept = -log(A)
    
    return k, A, r_squared


def plot_results(lambdas, k, A, r_squared, tick_size):
    """Plot calibration results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"K Parameter Calibration (tick_size={tick_size:.2e})", fontsize=14, fontweight="bold")
    
    # Filter valid data
    distances = np.array([d for d, l in lambdas.items() if l is not None])
    lambda_values = np.array([l for l in lambdas.values() if l is not None])
    
    if len(distances) == 0:
        print("No valid data to plot")
        return
    
    # 1. λ vs distance
    ax1 = axes[0, 0]
    ax1.bar(distances, lambda_values, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Distance from Mid (ticks)')
    ax1.set_ylabel('λ (seconds per fill)')
    ax1.set_title('Fill Rate vs Distance')
    ax1.grid(True, alpha=0.3)
    
    # 2. log(λ) vs distance with regression line
    ax2 = axes[0, 1]
    log_lambdas = np.log(lambda_values)
    ax2.scatter(distances, log_lambdas, s=50, alpha=0.7, color='red', label='Data')
    
    if k is not None:
        fit_line = np.log(1/A) + k * distances  # log(λ) = -log(A) + k*δ
        ax2.plot(distances, fit_line, 'b-', linewidth=2, label=f'Fit: log(λ) = {np.log(1/A):.2f} + {k:.4f}×δ')
        ax2.legend()
    
    ax2.set_xlabel('Distance from Mid (ticks)')
    ax2.set_ylabel('log(Time per Fill)')
    ax2.set_title(f'Log-Linear Fit (R² = {r_squared:.4f})' if r_squared else 'Log-Linear Fit')
    ax2.grid(True, alpha=0.3)
    
    # 3. Fill probability vs distance
    ax3 = axes[1, 0]
    fill_probs = 1.0 / lambda_values if len(lambda_values) > 0 else np.array([])
    ax3.plot(distances, fill_probs, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Distance from Mid (ticks)')
    ax3.set_ylabel('Fill Probability (1/λ)')
    ax3.set_title('Fill Probability Decay')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals
    ax4 = axes[1, 1]
    if k is not None:
        predicted = np.log(1/A) + k * distances
        residuals = log_lambdas - predicted
        ax4.scatter(distances, residuals, s=50, alpha=0.7, color='purple')
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Distance from Mid (ticks)')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Fit Residuals')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_results(k, A, r_squared, lambdas, tick_size):
    """Print calibration results."""
    print("\n" + "=" * 70)
    print("K PARAMETER CALIBRATION RESULTS")
    print("=" * 70)
    
    print(f"\nTick Size: {tick_size:.2e}")
    
    if k is not None:
        print(f"\nFitted Parameters:")
        print(f"  k = {k:.6f}")
        print(f"  A = {A:.6f} (fill intensity at δ=0)")
        print(f"  R² = {r_squared:.4f}")
        
        print(f"\nFill intensity model: λ(δ) = {A:.4f} × exp(-{k:.6f} × δ)")
        print(f"Time per fill model: t(δ) = {1/A:.4f} × exp({k:.6f} × δ) seconds")
        
        print(f"\nFill rates at different distances:")
        for tick_dist in sorted(lambdas.keys())[:10]:
            lam = lambdas[tick_dist]
            if lam is not None:
                fill_intensity = 1.0 / lam
                print(f"  δ={tick_dist:2d} ticks: t={lam:8.4f} s/fill, intensity={fill_intensity:.2f} fills/s")
    else:
        print("\nCould not fit k parameter (insufficient data)")
    
    print("=" * 70)


def main():
    base_dir = Path(__file__).parent.parent
    
    lob_path = base_dir / "data" / "cmf" / "lob.parquet"
    trades_path = base_dir / "data" / "cmf" / "trades.parquet"
    
    if not lob_path.exists() or not trades_path.exists():
        print(f"Error: Parquet files not found!")
        return
    
    # Load data
    lob_table, trades_table = load_data(lob_path, trades_path, sample_rows=500_000)
    
    # Extract columns
    data = extract_columns(lob_table, trades_table)
    
    # Calculate tick size
    tick_size = calculate_tick_size(data['lob_bid'], data['lob_ask'])
    print(f"Calculated tick size: {tick_size:.2e}")
    
    # Simulate order placement
    resting_time, fills = simulate_order_placement(data, tick_size, max_ticks=20)
    
    # Calculate λ
    lambdas = calculate_lambda(resting_time, fills)
    
    # Fit k parameter
    k, A, r_squared = fit_k_parameter(lambdas)
    
    # Print results
    print_results(k, A, r_squared, lambdas, tick_size)
    
    # Plot
    print("\nGenerating plots...")
    plot_results(lambdas, k, A, r_squared, tick_size)
    
    # Save results
    results_dir = base_dir / "results" / "k_calibration"
    results_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    import csv
    csv_path = results_dir / f"k_calibration_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tick_distance', 'resting_time_sec', 'fills', 'lambda', 'log_lambda'])
        for tick_dist in sorted(lambdas.keys()):
            lam = lambdas[tick_dist]
            writer.writerow([
                tick_dist,
                resting_time[tick_dist] / 1_000_000,
                fills[tick_dist],
                lam if lam else '',
                np.log(lam) if lam else ''
            ])
    
    print(f"\nResults saved to: {csv_path}")
    
    if k is not None:
        print(f"\nRecommended k for config: {k:.6f}")


if __name__ == "__main__":
    main()
