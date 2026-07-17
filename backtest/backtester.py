from dataclasses import dataclass

import pandas as pd

from indicators.signal_engine import compute_signal_series


@dataclass
class Trade:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float

    @property
    def pnl(self) -> float:
        move = self.exit_price - self.entry_price
        return move if self.direction == "BUY" else -move


def run_backtest(
    df: pd.DataFrame,
    weights: dict,
    thresholds: dict,
    max_hold_bars: int = 20,
) -> tuple[list[Trade], pd.Series]:
    """Walk forward over precomputed (causal) signals, one trade open at a time.

    Opens on BUY/SELL, closes on an opposite signal or after max_hold_bars, whichever
    comes first. Uses only past+current bar info at every step (no look-ahead).
    """
    scored = compute_signal_series(df, weights, thresholds)

    trades: list[Trade] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []
    running_pnl = 0.0
    open_trade: Trade | None = None
    bars_held = 0

    for ts, row in scored.iterrows():
        signal = row["signal"]
        price = row["close"]

        if open_trade is None:
            if signal in ("BUY", "SELL"):
                open_trade = Trade(direction=signal, entry_time=ts, entry_price=price, exit_time=ts, exit_price=price)
                bars_held = 0
        else:
            bars_held += 1
            opposite = signal != "HOLD" and signal != open_trade.direction
            if opposite or bars_held >= max_hold_bars:
                open_trade.exit_time = ts
                open_trade.exit_price = price
                running_pnl += open_trade.pnl
                trades.append(open_trade)
                open_trade = None
                bars_held = 0
                if opposite:
                    open_trade = Trade(direction=signal, entry_time=ts, entry_price=price, exit_time=ts, exit_price=price)
                    bars_held = 0

        equity_points.append((ts, running_pnl))

    if open_trade is not None:
        # Close out any still-open position at the last available bar so its PnL is counted.
        last_ts, last_price = scored.index[-1], scored["close"].iloc[-1]
        open_trade.exit_time = last_ts
        open_trade.exit_price = last_price
        running_pnl += open_trade.pnl
        trades.append(open_trade)
        equity_points[-1] = (last_ts, running_pnl)

    equity_curve = pd.Series(
        [pnl for _, pnl in equity_points],
        index=[ts for ts, _ in equity_points],
        name="equity",
    )
    return trades, equity_curve
