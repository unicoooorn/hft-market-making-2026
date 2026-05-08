#!/usr/bin/env python3
"""Grid search for Avellaneda-Stoikov hyperparameters."""

import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pyarrow.parquet as pq


@dataclass
class GridResult:
    gamma: float
    k: float
    order_volume: int
    cancellation_threshold: float
    horizon_seconds: float
    pnl: float
    equity: float
    turnover: float
    cancellations: int
    trades: int


def create_subset_data(
    base_dir: Path,
    lob_rows: int = 100_000,
    trades_rows: int = 2_000_000,
) -> tuple[Path, Path]:
    subset_dir = base_dir / "data" / "cmf" / "subset"
    subset_dir.mkdir(exist_ok=True)

    lob_path = base_dir / "data" / "cmf" / "lob.parquet"
    trades_path = base_dir / "data" / "cmf" / "trades.parquet"

    lob_subset_path = subset_dir / "lob.parquet"
    trades_subset_path = subset_dir / "trades.parquet"

    print(f"Creating LOB subset ({lob_rows:,} rows)...")
    lob_table = pq.read_table(lob_path)
    lob_subset = lob_table.slice(lob_table.num_rows - lob_rows, lob_table.num_rows) if lob_table.num_rows > lob_rows else lob_table
    pq.write_table(lob_subset, lob_subset_path, compression="snappy")

    print(f"Creating trades subset ({trades_rows:,} rows)...")
    trades_table = pq.read_table(trades_path)
    trades_subset = trades_table.slice(trades_table.num_rows - trades_rows, trades_table.num_rows) if trades_table.num_rows > trades_rows else trades_table
    pq.write_table(trades_subset, trades_subset_path, compression="snappy")

    print(f"Subset created: {lob_subset_path} ({lob_subset.num_rows:,} rows), {trades_subset_path} ({trades_subset.num_rows:,} rows)")
    return lob_subset_path, trades_subset_path


def run_backtest(
    params: tuple[float, float, int, float, float],
    lob_path: Path,
    trades_path: Path,
    batch_size: int = 100_000,
    verbose: bool = True,
) -> GridResult:
    gamma, k, order_volume, cancellation_threshold, horizon_seconds = params

    if verbose:
        print(f"  Running: gamma={gamma:.10f}, k={k:.10f}, vol={order_volume/1e8:.0f}, cancel_thr={cancellation_threshold:.10f}, horizon={horizon_seconds:.2f}s")

    from backtester import (
        Backtester,
        CSVDataProvider,
        LimitOrderExecutor,
        Inventory,
        AvellanedaStoikovStrategy,
        AvellanedaStoikovConfig,
    )

    try:
        inventory = Inventory()

        config = AvellanedaStoikovConfig(
            gamma=gamma,
            k=k,
            volatility_window=1000,
            horizon_seconds=horizon_seconds,
            min_spread=0.0000001,
            max_spread=0.000001,
            tick_size=0.00000001,
            order_amount=order_volume,
            cancellation_threshold=cancellation_threshold,
        )

        strategy = AvellanedaStoikovStrategy(
            inventory=inventory,
            config=config,
            strategy_id=f"as_{gamma}_{k}",
        )

        executor = LimitOrderExecutor()

        data_provider = CSVDataProvider(
            order_book_path=lob_path,
            trades_path=trades_path,
            batch_size=batch_size,
            show_progress=False,
        )

        backtester = Backtester(
            data_provider=data_provider,
            strategy=strategy,
            executor=executor,
            inventory=inventory,
        )

        backtester.run()

        chart_data = inventory.get_chart_data()
        pnl = chart_data["pnl"][-1] if chart_data["pnl"] else 0.0
        equity = chart_data["equity"][-1] if chart_data["equity"] else 0.0
        turnover = chart_data["turnover"][-1] if chart_data["turnover"] else 0.0

        result = GridResult(
            gamma=gamma,
            k=k,
            order_volume=order_volume,
            cancellation_threshold=cancellation_threshold,
            horizon_seconds=horizon_seconds,
            pnl=pnl,
            equity=equity,
            turnover=turnover,
            cancellations=executor.cancelled_count,
            trades=len(chart_data.get("timestamps", [])),
        )

        if verbose:
            print(f"    -> PnL: ${pnl:>10.2f}, Equity: ${equity:>8.2f}, Cancellations: {result.cancellations:,}")

        return result

    except Exception as e:
        if verbose:
            print(f"    -> ERROR: {e}")
        return GridResult(
            gamma=gamma,
            k=k,
            order_volume=order_volume,
            cancellation_threshold=cancellation_threshold,
            horizon_seconds=horizon_seconds,
            pnl=float("-inf"),
            equity=0.0,
            turnover=0.0,
            cancellations=0,
            trades=0,
        )


def grid_search(
    lob_path: Path,
    trades_path: Path,
    gamma_values: list[float],
    k_values: list[float],
    order_volume_values: list[int],
    cancellation_threshold_values: list[float],
    horizon_seconds_values: list[float],
    n_workers: int | None = None,
) -> list[GridResult]:
    if n_workers is None:
        n_workers = mp.cpu_count()

    param_grid = list(product(
        gamma_values,
        k_values,
        order_volume_values,
        cancellation_threshold_values,
        horizon_seconds_values,
    ))

    total_runs = len(param_grid)
    print(f"\n{'='*60}")
    print(f"GRID SEARCH")
    print(f"{'='*60}")
    print(f"Total combinations: {total_runs:,}")
    print(f"Workers: {n_workers}")
    print(f"Parameters:")
    print(f"  gamma: {gamma_values}")
    print(f"  k: {k_values}")
    print(f"  order_volume: {order_volume_values}")
    print(f"  cancellation_threshold: {cancellation_threshold_values}")
    print(f"  horizon_seconds: {horizon_seconds_values}")
    print(f"{'='*60}\n")

    args = [
        (params, lob_path, trades_path, 100_000, True)
        for params in param_grid
    ]

    results: list[GridResult] = []
    completed = 0

    with mp.Pool(processes=n_workers) as pool:
        for result in pool.starmap(run_backtest, args):
            results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == total_runs:
                print(f"\nProgress: {completed}/{total_runs} ({100*completed/total_runs:.1f}%)")

    results.sort(key=lambda r: r.pnl, reverse=True)

    return results


def print_top_results(results: list[GridResult], top_n: int = 20) -> None:
    print(f"\n{'='*100}")
    print(f"TOP {top_n} RESULTS (by PnL)")
    print(f"{'='*100}")
    print(f"{'Rank':<5} {'PnL':>12} {'Gamma':>14} {'K':>14} {'Order Vol':>12} {'Cancel Thr':>14} {'Horizon':>10} {'Cancels':>10}")
    print(f"{'-'*100}")

    for i, r in enumerate(results[:top_n], 1):
        print(f"{i:<5} {r.pnl:>12.2f} {r.gamma:>14.10f} {r.k:>14.10f} {r.order_volume:>12,.0f} {r.cancellation_threshold:>14.10f} {r.horizon_seconds:>10.2f} {r.cancellations:>10,}")

    print(f"{'='*100}")


def save_results(results: list[GridResult], output_path: Path) -> None:
    import csv

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "gamma", "k", "order_volume", "cancellation_threshold", "horizon_seconds", "pnl", "equity", "turnover", "cancellations", "trades"])
        for i, r in enumerate(results, 1):
            writer.writerow([
                i, r.gamma, r.k, r.order_volume, r.cancellation_threshold, r.horizon_seconds,
                r.pnl, r.equity, r.turnover, r.cancellations, r.trades
            ])

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    base_dir = Path(__file__).parent.parent

    print("Creating subset data...")
    lob_subset, trades_subset = create_subset_data(
        base_dir,
        lob_rows=2_000_000,
        trades_rows=30_000_000,
    )

    gamma_values = [0.007]
    k_values = [50_000_000, 500_000_000, 5_000_000_000]
    order_volume_values = [50 * 1e8, 500 * 1e8, 1000 * 1e8, 2000 * 1e8]
    cancellation_threshold_values = [0.1]
    horizon_seconds_values = [27]

    results = grid_search(
        lob_path=lob_subset,
        trades_path=trades_subset,
        gamma_values=gamma_values,
        k_values=k_values,
        order_volume_values=order_volume_values,
        cancellation_threshold_values=cancellation_threshold_values,
        horizon_seconds_values=horizon_seconds_values,
        n_workers=12,
    )

    print_top_results(results, top_n=20)

    results_dir = base_dir / "results" / "grid_search"
    results_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"grid_search_{timestamp}.csv"
    save_results(results, output_path)

    best = results[0]
    print(f"\n{'='*60}")
    print(f"BEST PARAMETERS:")
    print(f"  gamma: {best.gamma:.10f}")
    print(f"  k: {best.k:.10f}")
    print(f"  order_volume: {best.order_volume:,.0f}")
    print(f"  cancellation_threshold: {best.cancellation_threshold:.10f}")
    print(f"  horizon_seconds: {best.horizon_seconds:.2f}")
    print(f"  PnL: ${best.pnl:,.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
