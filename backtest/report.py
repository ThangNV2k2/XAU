import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def summarize(trades: list, equity_curve: pd.Series) -> dict:
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "net_pnl": 0.0,
        }

    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_rate = len(wins) / len(pnls) * 100
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    returns = pd.Series(pnls)
    sharpe = float(returns.mean() / returns.std() * (len(returns) ** 0.5)) if returns.std() > 0 else 0.0

    return {
        "total_trades": len(pnls),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else profit_factor,
        "max_drawdown": round(max_drawdown, 2),
        "sharpe": round(sharpe, 2),
        "net_pnl": round(sum(pnls), 2),
    }


def plot_equity_curve(equity_curve: pd.Series, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    equity_curve.plot(ax=ax)
    ax.set_title("Equity Curve (Backtest) - XAU/USD")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative PnL (USD/oz)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
