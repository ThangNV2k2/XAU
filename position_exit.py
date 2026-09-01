"""Closed-candle invalidation rules for an already open position.

Hard SL/TP checks belong to the execution layer because they use live Bid/Ask.
This module only decides whether the original setup has stopped being valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import pandas as pd

from strategy_engine import TechnicalSnapshot


@dataclass
class PositionState:
    symbol: str
    asset_type: str
    side: str
    setup: str
    entry_time: datetime
    entry_price: float
    initial_stop: float
    current_stop: float
    take_profit_1: float
    take_profit_2: float
    invalidation_level: float
    initial_risk: float
    remaining_fraction: float = 1.0
    realized_r: float = 0.0
    tp1_hit: bool = False
    bars_held_5m: int = 0
    max_favorable_r: float = 0.0


@dataclass(frozen=True)
class ExitDecision:
    action: str
    reason: str
    new_stop: float | None = None


def _is_broken(close: float, level: float, side: str) -> bool:
    return close < level if side == "LONG" else close > level


def assess_closed_candle_exit(
    position: PositionState,
    frames: Mapping[str, pd.DataFrame],
    snapshots: Mapping[str, TechnicalSnapshot],
    settings: Mapping[str, object] | None = None,
) -> ExitDecision:
    """Assess setup invalidation using completed candles only.

    A wick through the retest level is deliberately insufficient. Structure
    needs either two 5-minute closes or one 15-minute close beyond the fixed
    invalidation level, plus adverse RSI/score confirmation. A separate
    two-of-three 15m/1H/4H flip protects against a broader regime reversal.
    """
    settings = settings or {}
    frame_5m = frames.get("5min")
    snapshot_5m = snapshots.get("5min")
    if frame_5m is None or snapshot_5m is None or len(frame_5m) < 2:
        return ExitDecision("HOLD", "Chưa đủ nến đóng để kiểm tra phá vỡ")

    last_two = frame_5m["close"].astype(float).tail(2)
    two_5m_closes = len(last_two) == 2 and all(
        _is_broken(float(value), position.invalidation_level, position.side)
        for value in last_two
    )
    frame_15m = frames.get("15min")
    one_15m_close = bool(
        frame_15m is not None
        and not frame_15m.empty
        and _is_broken(
            float(frame_15m["close"].iloc[-1]),
            position.invalidation_level,
            position.side,
        )
    )

    score_floor = float(settings.get("adverse_score_floor", 0.08))
    if position.side == "LONG":
        adverse_momentum = bool(
            (snapshot_5m.rsi < 45.0 and snapshot_5m.rsi_slope < 0.0)
            or snapshot_5m.score <= -score_floor
        )
    else:
        adverse_momentum = bool(
            (snapshot_5m.rsi > 55.0 and snapshot_5m.rsi_slope > 0.0)
            or snapshot_5m.score >= score_floor
        )
    if (two_5m_closes or one_15m_close) and adverse_momentum:
        confirmation = "2 nến 5m" if two_5m_closes else "1 nến 15m"
        return ExitDecision(
            "FULL_EXIT",
            f"STRUCTURE_INVALIDATED: {confirmation} đóng qua mức {position.invalidation_level:g} và động lượng đảo",
        )

    alignment_floor = float(settings.get("alignment_floor", 0.08))
    higher = [snapshots[key].score for key in ("15min", "1h", "4h") if key in snapshots]
    if position.side == "LONG":
        opposing = sum(score <= -alignment_floor for score in higher)
    else:
        opposing = sum(score >= alignment_floor for score in higher)
    if len(higher) >= 2 and opposing >= 2:
        return ExitDecision(
            "FULL_EXIT",
            "REGIME_FLIP: ít nhất 2 khung 15m/1H/4H đã đồng thuận ngược vị thế",
        )

    time_stop_bars = max(1, int(settings.get("time_stop_bars_5m", 24)))
    minimum_progress_r = float(settings.get("minimum_progress_r", 0.50))
    direction_score = snapshot_5m.score if position.side == "LONG" else -snapshot_5m.score
    if (
        position.bars_held_5m >= time_stop_bars
        and position.max_favorable_r < minimum_progress_r
        and direction_score <= 0.0
    ):
        return ExitDecision(
            "FULL_EXIT",
            f"TIME_STOP: sau {position.bars_held_5m * 5} phút chưa đạt {minimum_progress_r:g}R và động lượng mất",
        )

    if position.tp1_hit and (
        position.side == "LONG" and position.current_stop < position.entry_price
        or position.side == "SHORT" and position.current_stop > position.entry_price
    ):
        return ExitDecision("MOVE_STOP", "TP1 đã đạt: dời SL phần còn lại về hòa vốn", position.entry_price)
    return ExitDecision("HOLD", "Setup còn hiệu lực")
