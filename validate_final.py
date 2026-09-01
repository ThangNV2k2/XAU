"""Kiem chung cau hinh cuoi tren tick Exness that, tach in-sample / out-of-sample.

Muc dich: chung minh cac thay doi khong phai curve-fit. Tham so duoc chon
tren nua dau; nua sau la du lieu chien luoc chua he "nhin thay" luc chon.
"""
import copy, yaml, pandas as pd
from backtest.replay import replay_strategy

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
bars = pd.read_pickle("backtest/cache/XAUUSD_2025_2026_1m.pkl")
PERIODS = [
    ("IN-SAMPLE  2025-06 -> 2026-02", "2025-06-01", "2026-02-28"),
    ("OUT-SAMPLE 2026-03 -> 2026-08", "2026-03-01", None),
    ("TOAN BO    2025-06 -> 2026-08", "2025-06-01", None),
]
CONFIGS = {
    "1. GOC (5m, khong loc thien huong)": {
        "entry_interval": "5min", "actionable_score": 0.30,
        "alignment_floor": 0.08, "require_bias_alignment": False},
    "2. + entry 15m": {
        "entry_interval": "15min", "actionable_score": 0.30,
        "alignment_floor": 0.08, "require_bias_alignment": False},
    "3. + siet score/alignment": {
        "entry_interval": "15min", "actionable_score": 0.40,
        "alignment_floor": 0.20, "require_bias_alignment": False},
    "4. + loc thien huong D1  <-- config.yaml": {},
}
with open("logs/validate_final.txt", "w", encoding="utf-8") as out:
    for name, override in CONFIGS.items():
        settings = copy.deepcopy(cfg["strategy"]); settings.update(override)
        print(f"\n### {name}", file=out, flush=True)
        for label, start, end in PERIODS:
            result = replay_strategy(
                bars, symbol="XAUUSD", asset_type="metal", strategy_settings=settings,
                exit_settings=cfg["position_exit"], digits=3,
                entry_cooldown_minutes=cfg["alerts"]["signal_cooldown_minutes"],
                evaluation_start=start, evaluation_end=end)
            x = result["summaries"]["baseline"]
            pf = x["profit_factor"] if x["profit_factor"] is not None else "inf"
            print(f"  {label}: {x['trades']:4d} lenh  net {x['net_r']:+7.2f}R  "
                  f"win {x['win_rate']:5.1f}%  PF {pf}  MDD {x['max_drawdown_r']:.1f}R",
                  file=out, flush=True)
