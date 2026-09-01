"""Chấm điểm chất lượng entry của replay trên nến Exness thật.

Trả lời một câu hỏi duy nhất: lệnh vào có đứng ở chỗ tử tế không, hay chỉ là
đúng hướng nhờ may. Dùng MAE/MFE trên chính nến Bid/Ask 1m đã sinh ra lệnh.
"""

from __future__ import annotations

import sys

import pandas as pd


def excursions(bars: pd.DataFrame, trade: dict) -> dict:
    """MAE/MFE tính bằng R, trên đúng khoảng thời gian lệnh còn mở."""
    entry = pd.Timestamp(trade["entry_time"])
    exit_at = pd.Timestamp(trade["exit_time"])
    window = bars.loc[entry:exit_at]
    if window.empty:
        return {"mae_r": 0.0, "mfe_r": 0.0, "bars": 0}
    price = float(trade["entry_price"])
    risk = abs(price - float(trade["initial_stop"])) or 1e-9
    if trade["side"] == "LONG":
        best, worst = window["bid_high"].max(), window["bid_low"].min()
        mfe, mae = (best - price) / risk, (worst - price) / risk
    else:
        best, worst = window["ask_low"].min(), window["ask_high"].max()
        mfe, mae = (price - best) / risk, (price - worst) / risk
    return {"mae_r": float(mae), "mfe_r": float(mfe), "bars": int(len(window))}


def entry_quality(bars: pd.DataFrame, trades: list[dict]) -> pd.DataFrame:
    rows = []
    for trade in trades:
        row = dict(trade)
        row.update(excursions(bars, trade))
        entry = pd.Timestamp(trade["entry_time"])
        row["hour_utc"] = entry.hour
        row["risk_pct"] = abs(
            float(trade["entry_price"]) - float(trade["initial_stop"])
        ) / float(trade["entry_price"]) * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def report(frame: pd.DataFrame, variant: str) -> None:
    data = frame[frame["variant"] == variant]
    if data.empty:
        print(f"\n### {variant}: khong co lenh nao")
        return
    wins = data[data["realized_r"] > 0]
    losses = data[data["realized_r"] <= 0]
    print(f"\n### {variant} — {len(data)} lenh")
    print(f"  net {data['realized_r'].sum():+.2f}R | trung binh {data['realized_r'].mean():+.3f}R "
          f"| win {len(wins) / len(data) * 100:.1f}%")
    print(f"  MAE trung vi {data['mae_r'].median():+.2f}R | MFE trung vi {data['mfe_r'].median():+.2f}R")
    print(f"  MAE cua lenh THANG: {wins['mae_r'].median():+.2f}R" if len(wins) else "  (khong co lenh thang)")
    print(f"  MFE cua lenh THUA : {losses['mfe_r'].median():+.2f}R" if len(losses) else "  (khong co lenh thua)")
    # Entry duoc coi la "khong bi thoc" neu gia chua bao gio di nguoc qua 0.5R
    clean = float((data["mae_r"] > -0.5).mean() * 100)
    never_worked = float((data["mfe_r"] < 0.5).mean() * 100)
    print(f"  Entry sach (MAE > -0.5R): {clean:.1f}%")
    print(f"  Chua bao gio chay (MFE < 0.5R): {never_worked:.1f}%")
    for side in ("LONG", "SHORT"):
        part = data[data["side"] == side]
        if len(part):
            print(f"  {side}: {len(part)} lenh, net {part['realized_r'].sum():+.2f}R, "
                  f"win {float((part['realized_r'] > 0).mean() * 100):.1f}%")
    print(f"  Ly do thoat: {data['exit_reason'].str.split(':').str[0].value_counts().to_dict()}")
    print(f"  Setup: {data['setup'].value_counts().to_dict()}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json

    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        result = json.load(handle)
    bars = pd.read_pickle(sys.argv[2])
    trades = result["trades"]
    if not trades:
        print("Replay khong sinh lenh nao.")
        print(json.dumps(result["summaries"], ensure_ascii=False, indent=2))
        return
    frame = entry_quality(bars, trades)
    frame.to_csv("logs/entry_quality.csv", index=False, encoding="utf-8")
    print(f"Nguon: {result['start']} -> {result['end']} | {result['signal_events']} su kien tin hieu")
    for variant in ("baseline", "managed_exit"):
        report(frame, variant)
    print("\nChi tiet: logs/entry_quality.csv")


if __name__ == "__main__":
    main()
