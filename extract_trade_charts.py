"""Trích nến 5m quanh từng lệnh replay để vẽ lại đúng bối cảnh vào lệnh."""

from __future__ import annotations

import json
import sys

import pandas as pd

from backtest.replay import build_timeframes
from strategy_engine import technical_frame


BEFORE_BARS = 48
AFTER_BARS = 12


def candles_for(trade: dict, frame5: pd.DataFrame, tech5: pd.DataFrame) -> dict | None:
    entry = pd.Timestamp(trade["entry_time"])
    exit_at = pd.Timestamp(trade["exit_time"])
    start_position = frame5.index.searchsorted(entry, side="right") - 1
    end_position = frame5.index.searchsorted(exit_at, side="right") - 1
    if start_position < BEFORE_BARS:
        return None
    low = max(0, start_position - BEFORE_BARS)
    high = min(len(frame5) - 1, end_position + AFTER_BARS)
    window = frame5.iloc[low : high + 1]
    technical = tech5.iloc[low : high + 1]
    return {
        "side": trade["side"],
        "setup": trade["setup"],
        "entry_time": str(entry),
        "exit_time": str(exit_at),
        "entry": float(trade["entry_price"]),
        "stop": float(trade["initial_stop"]),
        "invalidation": float(trade["invalidation_level"]),
        "exit_price": float(trade["exit_price"]),
        "realized_r": float(trade["realized_r"]),
        "exit_reason": trade["exit_reason"],
        "entry_index": int(start_position - low),
        "exit_index": int(end_position - low),
        "candles": [
            {
                "t": str(index),
                "o": round(float(row["open"]), 2),
                "h": round(float(row["high"]), 2),
                "l": round(float(row["low"]), 2),
                "c": round(float(row["close"]), 2),
            }
            for index, row in window.iterrows()
        ],
        "ema20": [round(float(value), 2) for value in technical["ema20"]],
        "ema50": [round(float(value), 2) for value in technical["ema50"]],
        "rsi": [round(float(value), 1) for value in technical["rsi"]],
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = json.load(open(sys.argv[1], "r", encoding="utf-8"))
    bars = pd.read_pickle(sys.argv[2])
    frames = build_timeframes(bars)
    frame5 = frames["5min"]
    tech5 = technical_frame(frame5)

    trades = [item for item in result["trades"] if item["variant"] == "baseline"]
    trades.sort(key=lambda item: item["realized_r"])
    chosen = {
        "thua_nang_nhat": trades[:3],
        "thang_tot_nhat": trades[-3:][::-1],
        "dien_hinh": trades[len(trades) // 2 - 1 : len(trades) // 2 + 2],
    }
    output = {}
    for label, group in chosen.items():
        rendered = [candles_for(item, frame5, tech5) for item in group]
        output[label] = [item for item in rendered if item is not None]
    json.dump(output, open("logs/trade_charts.json", "w", encoding="utf-8"), ensure_ascii=False)
    for label, group in output.items():
        print(f"{label}: {len(group)} bieu do")


if __name__ == "__main__":
    main()
