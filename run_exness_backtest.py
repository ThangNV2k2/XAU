"""CLI for an Exness-only historical strategy replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

from backtest.exness_tick_history import download_archive, ticks_to_one_minute
from backtest.replay import replay_strategy


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Replay chiến lược trên Tick History Bid/Ask của Exness")
    parser.add_argument("--symbol", default="XAUUSD", help="Đúng mã/suffix Exness, ví dụ XAUUSD hoặc XAUUSDm")
    parser.add_argument("--asset-type", choices=("metal", "stock"), default="metal")
    parser.add_argument("--input", action="append", default=[], help="ZIP/CSV Tick History; dùng nhiều lần được")
    parser.add_argument("--period", action="append", default=[], help="Tự tải YYYY, YYYY-MM hoặc YYYY-MM-DD")
    parser.add_argument("--cache-dir", default="backtest/cache")
    parser.add_argument("--bars-cache", default="", help="File pickle 1m để không phải aggregate lại")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--evaluation-start", default="", help="Chỉ mở lệnh từ thời điểm UTC này; dữ liệu trước vẫn warm-up")
    parser.add_argument("--evaluation-end", default="", help="Dừng mở/mô phỏng tại thời điểm UTC này")
    parser.add_argument("--allow-missing-ema200", action="store_true", help="Chỉ dùng để chẩn đoán mẫu quá ngắn")
    parser.add_argument("--output", default="logs/exness_backtest.json")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    paths = [Path(item) for item in args.input]
    for period in args.period:
        paths.append(download_archive(args.symbol, period, args.cache_dir))
    if not paths and not args.bars_cache:
        parser.error("Cần ít nhất một --input/--period hoặc --bars-cache")

    cache_path = Path(args.bars_cache) if args.bars_cache else None
    if cache_path is not None and cache_path.exists():
        bars = pd.read_pickle(cache_path)
        if paths:
            additions = ticks_to_one_minute(paths)
            bars = pd.concat([bars, additions]).sort_index(kind="stable")
            bars = bars[~bars.index.duplicated(keep="last")]
            bars.to_pickle(cache_path)
    else:
        bars = ticks_to_one_minute(paths)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            bars.to_pickle(cache_path)

    config = _load_config(args.config)
    result = replay_strategy(
        bars,
        symbol=args.symbol,
        asset_type=args.asset_type,
        strategy_settings=config.get("strategy", {}),
        exit_settings=config.get("position_exit", {}),
        digits=args.digits,
        require_ema200=not args.allow_missing_ema200,
        entry_cooldown_minutes=int(config.get("alerts", {}).get("signal_cooldown_minutes", 60)),
        evaluation_start=args.evaluation_start or None,
        evaluation_end=args.evaluation_end or None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"Nguồn: {result['source_bars_1m']} nến 1m · {result['start']} → {result['end']}")
    print(f"Sự kiện tín hiệu: {result['signal_events']}")
    for summary in result["summaries"].values():
        factor = summary["profit_factor"] if summary["profit_factor"] is not None else "∞"
        print(
            f"{summary['variant']}: {summary['trades']} lệnh · net {summary['net_r']:+.3f}R · "
            f"avg {summary['average_r']:+.3f}R · win {summary['win_rate']:.2f}% · "
            f"PF {factor} · MDD {summary['max_drawdown_r']:.3f}R"
        )
        print(f"  Exit: {summary['exit_reasons']}")
    print(f"Chi tiết: {output}")


if __name__ == "__main__":
    main()
