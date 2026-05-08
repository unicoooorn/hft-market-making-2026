#!/usr/bin/env python3
"""Profile the backtester to find performance bottlenecks."""

import cProfile
import pstats
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtester import (
    Backtester,
    CSVDataProvider,
    StubStrategy,
    LimitOrderExecutor,  # noqa: F401
)
from backtester.config import load_config


def run_backtest() -> None:
    """Run the backtest - called by profiler."""
    # Config path relative to project root, not script
    config_path = Path(__file__).parent.parent / "config" / "local.yaml"
    config = load_config(config_path)

    data_provider = CSVDataProvider(
        order_book_path=config.order_book_path,
        trades_path=config.trades_path,
        batch_size=config.batch_size,
        show_progress=False,  # Disable progress for profiling
    )

    strategy = StubStrategy(
        strategy_id=config.strategy_id,
        order_volume=config.order_volume,
    )
    executor = LimitOrderExecutor()

    backtester = Backtester(
        data_provider=data_provider,
        strategy=strategy,
        executor=executor,
    )

    backtester.run()
    print("Backtest completed successfully!")


def profile_output(sort_by: str = "cumulative", lines: int = 50) -> None:
    """Run profiler and print results."""
    profiler = cProfile.Profile()
    
    print("Starting profiler...")
    print("=" * 60)
    
    profiler.enable()
    run_backtest()
    profiler.disable()
    
    print(f"\n{'=' * 60}")
    print(f"PROFILING RESULTS (sorted by {sort_by})")
    print(f"{'=' * 60}\n")
    
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats(sort_by)
    stats.print_stats(lines)
    
    # Also show callers for top functions
    print(f"\n{'=' * 60}")
    print("CALLERS (who called the slowest functions)")
    print(f"{'=' * 60}\n")
    stats.print_callers(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile the backtester")
    parser.add_argument(
        "--sort",
        choices=["cumulative", "time", "calls", "filename", "name"],
        default="cumulative",
        help="Sort order for profiling results",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=50,
        help="Number of lines to show",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save raw profile data to file (.prof)",
    )
    
    args = parser.parse_args()
    
    if args.output:
        # Save raw profile data for later analysis with snakeviz/tuna
        profiler = cProfile.Profile()
        profiler.enable()
        run_backtest()
        profiler.disable()
        profiler.dump_stats(args.output)
        print(f"Profile data saved to: {args.output}")
        print(f"View with: snakeviz {args.output}")
    else:
        profile_output(sort_by=args.sort, lines=args.lines)
