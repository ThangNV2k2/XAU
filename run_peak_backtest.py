import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from backtest.binance_history import build_timeframes, fetch_binance_history
from backtest.peak_backtester import run_peak_backtest


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point-in-time replay of the live /dinh strategy on Binance XAUUSDT"
    )
    parser.add_argument("--start", default="2025-12-11T08:05:00Z")
    parser.add_argument("--end", default=datetime.now(timezone.utc).isoformat())
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--trades-output", default="logs/peak_backtest_trades.json")
    parser.add_argument("--summary-output", default="logs/peak_backtest_summary.json")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = fetch_binance_history(
        config.get("symbol", "XAUUSDT"),
        "15min",
        parse_utc(args.start),
        parse_utc(args.end),
        force_refresh=args.force_refresh,
    )
    print(f"Loaded {len(base)} Binance 15m bars: {base.index[0]} -> {base.index[-1]}")
    trades, stats = run_peak_backtest(build_timeframes(base), config)
    output_path = Path(args.trades_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(trade) for trade in trades], handle, ensure_ascii=False, indent=2, default=str)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Trades saved to {output_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
