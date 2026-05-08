from pathlib import Path
import shutil

from backtester import (
    Backtester,
    CSVDataProvider,
    LimitOrderExecutor,
    Inventory,
    plot_inventory_charts,
    AvellanedaStoikovStrategy,
    AvellanedaStoikovConfig,
)
from backtester.config import load_config


def save_experiment(
    chart_data: dict,
    quote_history: list[dict],
    config_path: Path,
    executor_cancelled_count: int,
) -> Path:
    """Save experiment results to timestamped folder.

    Returns:
        Path to the experiment folder
    """
    from datetime import datetime

    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving
    import matplotlib.pyplot as plt

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_folder = results_dir / timestamp
    exp_folder.mkdir(exist_ok=True)

    # Save figure
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("Backtest Results", fontsize=16, fontweight="bold")

    timestamps = chart_data["timestamps"]

    ax_equity = axes[0, 0]
    ax_equity.plot(timestamps, chart_data["equity"], "b-", linewidth=1.5)
    ax_equity.set_title("Equity", fontweight="bold")
    ax_equity.set_xlabel("Time")
    ax_equity.set_ylabel("USD")
    ax_equity.grid(True, alpha=0.3)

    ax_pnl = axes[0, 1]
    pnl_color = "green" if chart_data["pnl"][-1] >= 0 else "red"
    ax_pnl.plot(timestamps, chart_data["pnl"], color=pnl_color, linewidth=1.5)
    ax_pnl.set_title(f"PnL (Final: ${chart_data['pnl'][-1]:,.2f})", fontweight="bold")
    ax_pnl.set_xlabel("Time")
    ax_pnl.set_ylabel("USD")
    ax_pnl.grid(True, alpha=0.3)

    ax_usd = axes[1, 0]
    ax_usd.plot(timestamps, chart_data["usd"], "g-", linewidth=1.5)
    ax_usd.set_title("USD Balance", fontweight="bold")
    ax_usd.set_xlabel("Time")
    ax_usd.set_ylabel("USD")
    ax_usd.grid(True, alpha=0.3)

    ax_tokens = axes[1, 1]
    ax_tokens.plot(timestamps, chart_data["tokens"], "orange", linewidth=1.5)
    ax_tokens.set_title("Token Balance", fontweight="bold")
    ax_tokens.set_xlabel("Time")
    ax_tokens.set_ylabel("Tokens")
    ax_tokens.grid(True, alpha=0.3)

    ax_turnover = axes[2, 0]
    ax_turnover.plot(timestamps, chart_data["turnover"], "purple", linewidth=1.5)
    ax_turnover.set_title(f"Turnover (Final: ${chart_data['turnover'][-1]:,.2f})", fontweight="bold")
    ax_turnover.set_xlabel("Time")
    ax_turnover.set_ylabel("USD")
    ax_turnover.grid(True, alpha=0.3)

    ax_midprice = axes[2, 1]
    if quote_history:
        from datetime import datetime as dt
        step = max(1, len(quote_history) // 10000)
        sampled = quote_history[::step][:10000]
        q_timestamps = [dt.fromtimestamp(q["timestamp_us"] / 1_000_000) for q in sampled]
        mid_prices = [q["mid_price"] for q in sampled]
        ax_midprice.plot(q_timestamps, mid_prices, "cyan", linewidth=0.5, alpha=0.7)
        ax_midprice.set_title(f"Mid Price (samples: {len(mid_prices):,})", fontweight="bold")
        ax_midprice.set_xlabel("Time")
        ax_midprice.set_ylabel("Price")
        ax_midprice.grid(True, alpha=0.3)
    ax_midprice.tick_params(axis="x", rotation=45)

    ax_quotes = axes[3, 0]
    if quote_history:
        from datetime import datetime as dt
        step = max(1, len(quote_history) // 10000)
        sampled = quote_history[::step][:10000]
        q_timestamps = [dt.fromtimestamp(q["timestamp_us"] / 1_000_000) for q in sampled]
        mid_prices = [(q["best_bid"] + q["best_ask"]) / 2 for q in sampled]
        best_bid_diff = [q["best_bid"] - mid for q, mid in zip(sampled, mid_prices)]
        best_ask_diff = [q["best_ask"] - mid for q, mid in zip(sampled, mid_prices)]
        strat_bid_diff = [q["strategy_bid"] - mid if q["strategy_bid"] > 0 else None for q, mid in zip(sampled, mid_prices)]
        strat_ask_diff = [q["strategy_ask"] - mid if q["strategy_ask"] > 0 else None for q, mid in zip(sampled, mid_prices)]

        ax_quotes.axhline(y=0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax_quotes.plot(q_timestamps, best_bid_diff, "b-", linewidth=0.5, alpha=0.7, label="Best Bid - Mid")
        ax_quotes.plot(q_timestamps, best_ask_diff, "r-", linewidth=0.5, alpha=0.7, label="Best Ask - Mid")

        strat_bid_pts = [(q_timestamps[i], d) for i, d in enumerate(strat_bid_diff) if d is not None]
        strat_ask_pts = [(q_timestamps[i], d) for i, d in enumerate(strat_ask_diff) if d is not None]

        if strat_bid_pts:
            ax_quotes.scatter([t for t, _ in strat_bid_pts], [d for _, d in strat_bid_pts], c="green", s=1, alpha=0.5, label="Strategy Bid - Mid")
        if strat_ask_pts:
            ax_quotes.scatter([t for t, _ in strat_ask_pts], [d for _, d in strat_ask_pts], c="orange", s=1, alpha=0.5, label="Strategy Ask - Mid")

        ax_quotes.set_title("Quotes vs Mid Price", fontweight="bold")
        ax_quotes.set_xlabel("Time")
        ax_quotes.set_ylabel("Price Difference")
        ax_quotes.legend()
        ax_quotes.grid(True, alpha=0.3)
    ax_quotes.tick_params(axis="x", rotation=45)

    axes[3, 1].axis("off")
    plt.tight_layout()

    fig_path = exp_folder / "backtest_results.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Copy config
    config_dest = exp_folder / "local.yaml"
    shutil.copy2(config_path, config_dest)

    # Save summary
    summary_path = exp_folder / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Backtest Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Final USD balance: ${chart_data['usd'][-1]:,.2f}\n")
        f.write(f"Final Token balance: {chart_data['tokens'][-1]:,.2f}\n")
        f.write(f"Final Equity: ${chart_data['equity'][-1]:,.2f}\n")
        f.write(f"Total PnL: ${chart_data['pnl'][-1]:,.2f}\n")
        f.write(f"Total Turnover: ${chart_data['turnover'][-1]:,.2f}\n")
        f.write(f"Total snapshots: {len(chart_data['timestamps'])}\n")
        f.write(f"Total cancellations: {executor_cancelled_count}\n")

    print(f"\nExperiment saved to: {exp_folder}")
    return exp_folder


def main() -> None:
    """Run the backtester with Avellaneda-Stoikov market making strategy."""
    config_path = Path(__file__).parent / "config" / "local.yaml"
    config = load_config(config_path)

    data_provider = CSVDataProvider(
        order_book_path=config.order_book_path,
        trades_path=config.trades_path,
        batch_size=config.batch_size,
    )

    inventory = Inventory()

    as_cfg = config.as_config or AvellanedaStoikovConfig()
    strategy = AvellanedaStoikovStrategy(
        inventory=inventory,
        config=AvellanedaStoikovConfig(
            gamma=as_cfg.gamma,
            k=as_cfg.k,
            volatility_window=as_cfg.volatility_window,
            horizon_seconds=as_cfg.horizon_seconds,
            min_spread=as_cfg.min_spread,
            max_spread=as_cfg.max_spread,
            tick_size=as_cfg.tick_size,
            order_amount=config.order_volume,
            cancellation_threshold=as_cfg.cancellation_threshold,
        ),
        strategy_id="as_mm",
    )

    executor = LimitOrderExecutor()

    backtester = Backtester(
        data_provider=data_provider,
        strategy=strategy,
        executor=executor,
        inventory=inventory,
    )

    backtester.run()

    chart_data = inventory.get_chart_data()
    if chart_data["timestamps"]:
        print(f"\nBacktest completed!")
        print(f"Final USD balance: ${chart_data['usd'][-1]:,.2f}")
        print(f"Final Token balance: {chart_data['tokens'][-1]:,.2f}")
        print(f"Final Equity: ${chart_data['equity'][-1]:,.2f}")
        print(f"Total PnL: ${chart_data['pnl'][-1]:,.2f}")
        print(f"Total Turnover: ${chart_data['turnover'][-1]:,.2f}")
        print(f"Total snapshots: {len(chart_data['timestamps'])}")
        print(f"Total cancellations: {executor.cancelled_count}")

        quote_history = [
            {
                "timestamp_us": q.timestamp_us,
                "best_bid": q.best_bid,
                "best_ask": q.best_ask,
                "strategy_bid": q.strategy_bid,
                "strategy_ask": q.strategy_ask,
                "mid_price": q.mid_price,
            }
            for q in backtester.quote_history
        ]

        save_experiment(chart_data, quote_history, config_path, executor.cancelled_count)

        # Show interactive plots
        plot_inventory_charts(chart_data, quote_history)
    else:
        print("Backtest completed successfully (no data)!")


if __name__ == "__main__":
    main()
