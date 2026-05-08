"""Chart generation for backtest results."""

from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt


def plot_inventory_charts(
    chart_data: dict[str, list[Any]],
    quote_history: list[dict] | None = None,
    sample_size: int = 10000,
) -> None:
    """Generate and display charts from inventory data and optional quote history.

    Creates a 4x2 grid with:
    - Equity, PnL
    - USD Balance, Token Balance
    - Turnover, Mid Price
    - Quotes vs Mid (if quote_history provided)
    """
    if not chart_data["timestamps"]:
        print("No chart data available")
        return

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("Backtest Results", fontsize=16, fontweight="bold")

    timestamps = chart_data["timestamps"]

    ax_equity = axes[0, 0]
    ax_equity.plot(timestamps, chart_data["equity"], "b-", linewidth=1.5)
    ax_equity.set_title("Equity", fontweight="bold")
    ax_equity.set_xlabel("Time")
    ax_equity.set_ylabel("USD")
    ax_equity.grid(True, alpha=0.3)
    ax_equity.tick_params(axis="x", rotation=45)

    ax_pnl = axes[0, 1]
    pnl_color = "green" if chart_data["pnl"][-1] >= 0 else "red"
    ax_pnl.plot(timestamps, chart_data["pnl"], color=pnl_color, linewidth=1.5)
    ax_pnl.set_title(f"PnL (Final: ${chart_data['pnl'][-1]:,.2f})", fontweight="bold")
    ax_pnl.set_xlabel("Time")
    ax_pnl.set_ylabel("USD")
    ax_pnl.grid(True, alpha=0.3)
    ax_pnl.tick_params(axis="x", rotation=45)

    ax_usd = axes[1, 0]
    ax_usd.plot(timestamps, chart_data["usd"], "g-", linewidth=1.5)
    ax_usd.set_title("USD Balance", fontweight="bold")
    ax_usd.set_xlabel("Time")
    ax_usd.set_ylabel("USD")
    ax_usd.grid(True, alpha=0.3)
    ax_usd.tick_params(axis="x", rotation=45)

    ax_tokens = axes[1, 1]
    ax_tokens.plot(timestamps, chart_data["tokens"], "orange", linewidth=1.5)
    ax_tokens.set_title("Token Balance", fontweight="bold")
    ax_tokens.set_xlabel("Time")
    ax_tokens.set_ylabel("Tokens")
    ax_tokens.grid(True, alpha=0.3)
    ax_tokens.tick_params(axis="x", rotation=45)

    ax_turnover = axes[2, 0]
    ax_turnover.plot(timestamps, chart_data["turnover"], "purple", linewidth=1.5)
    ax_turnover.set_title(f"Turnover (Final: ${chart_data['turnover'][-1]:,.2f})", fontweight="bold")
    ax_turnover.set_xlabel("Time")
    ax_turnover.set_ylabel("USD")
    ax_turnover.grid(True, alpha=0.3)
    ax_turnover.tick_params(axis="x", rotation=45)

    ax_midprice = axes[2, 1]
    if quote_history:
        step = max(1, len(quote_history) // sample_size)
        sampled = quote_history[::step][:sample_size]
        q_timestamps = [datetime.fromtimestamp(q["timestamp_us"] / 1_000_000) for q in sampled]
        mid_prices = [q["mid_price"] for q in sampled]
        ax_midprice.plot(q_timestamps, mid_prices, "cyan", linewidth=0.5, alpha=0.7)
        ax_midprice.set_title(f"Mid Price (samples: {len(mid_prices):,})", fontweight="bold")
        ax_midprice.set_xlabel("Time")
        ax_midprice.set_ylabel("Price")
        ax_midprice.grid(True, alpha=0.3)
    else:
        ax_midprice.text(0.5, 0.5, "No midprice data", ha="center", va="center", fontsize=14)
        ax_midprice.axis("off")
    ax_midprice.tick_params(axis="x", rotation=45)

    ax_quotes = axes[3, 0]
    if quote_history:
        step = max(1, len(quote_history) // sample_size)
        sampled = quote_history[::step][:sample_size]

        q_timestamps = [datetime.fromtimestamp(q["timestamp_us"] / 1_000_000) for q in sampled]
        
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
            ax_quotes.scatter(
                [t for t, _ in strat_bid_pts],
                [d for _, d in strat_bid_pts],
                c="green",
                s=1,
                alpha=0.5,
                label="Strategy Bid - Mid",
            )
        if strat_ask_pts:
            ax_quotes.scatter(
                [t for t, _ in strat_ask_pts],
                [d for _, d in strat_ask_pts],
                c="orange",
                s=1,
                alpha=0.5,
                label="Strategy Ask - Mid",
            )

        ax_quotes.set_title("Quotes vs Mid Price", fontweight="bold")
        ax_quotes.set_xlabel("Time")
        ax_quotes.set_ylabel("Price Difference")
        ax_quotes.legend()
        ax_quotes.grid(True, alpha=0.3)
    else:
        ax_quotes.text(0.5, 0.5, "No quote data", ha="center", va="center", fontsize=14)
        ax_quotes.axis("off")
    ax_quotes.tick_params(axis="x", rotation=45)

    axes[3, 1].axis("off")

    plt.tight_layout()
    plt.show()
