import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


VIETNAM_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (TypeError, ValueError):
            pass
    return str(value)


def _frame_snapshot(frame, bars: int) -> dict:
    selected_columns = [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_quote_volume",
        )
        if column in frame.columns
    ]
    recent = []
    for timestamp, row in frame.tail(max(1, bars))[selected_columns].iterrows():
        recent.append(
            {
                "at": _json_safe(timestamp),
                **{
                    column: _json_safe(row[column])
                    for column in selected_columns
                },
            }
        )
    return {
        "closed_candle_at": _json_safe(frame.index[-1]) if not frame.empty else None,
        "bar_count": len(frame),
        "recent_closed_candles": recent,
    }


def _alignment_side(scores: dict[str, float], minimum_score: float) -> tuple[str, bool, bool]:
    required = ("15min", "1h", "4h")
    long_aligned = all(scores.get(item, 0.0) >= minimum_score for item in required)
    short_aligned = all(scores.get(item, 0.0) <= -minimum_score for item in required)
    side = "LONG" if long_aligned else "SHORT" if short_aligned else "MIXED"
    return side, long_aligned, short_aligned


def build_analysis_snapshot(
    opportunity,
    config: dict,
    phase: str | None,
    executable_price: float,
    event: str = "analysis_snapshot",
) -> dict:
    diagnostic_settings = config.get("analysis_diagnostics", {})
    signal_settings = config.get("signal_confirmation", {})
    execution_settings = config.get("peak_execution", {})
    liquidity_settings = config.get("peak_liquidity", {})
    auto_settings = config.get("auto_alerts", {})
    minimum_score = float(signal_settings.get("minimum_timeframe_score", 0.12))
    candidate_side, long_aligned, short_aligned = _alignment_side(
        opportunity.momentum_scores,
        minimum_score,
    )

    plan = opportunity.execution_plan
    gate = opportunity.gate
    liquidity = opportunity.liquidity
    trap = opportunity.trap
    macro_risk = opportunity.macro_risk
    quality = opportunity.quality
    resistance_exists = gate.resistance is not None
    pattern_confirmed = (
        gate.long_retest_confirmed
        if candidate_side == "LONG"
        else gate.short_rejection_confirmed
        if candidate_side == "SHORT"
        else False
    )
    entry_distance = None
    if plan is not None:
        if executable_price < plan.entry_lower:
            entry_distance = plan.entry_lower - executable_price
        elif executable_price > plan.entry_upper:
            entry_distance = executable_price - plan.entry_upper
        else:
            entry_distance = 0.0

    alignment_passed = long_aligned or short_aligned
    gate_direction_applicable = resistance_exists and liquidity.entries_allowed and alignment_passed
    execution_applicable = gate.allowed_decision in ("CANH LONG", "CANH SHORT")
    sizing_applicable = plan is not None
    proximity_applicable = opportunity.sized_plan is not None
    decision_path = [
        {
            "step": "resistance_zone_exists",
            "applicable": True,
            "passed": resistance_exists,
            "detail": gate.resistance,
        },
        {
            "step": "liquidity_and_spread",
            "applicable": resistance_exists,
            "passed": liquidity.entries_allowed,
            "detail": liquidity.reason,
        },
        {
            "step": "macro_event_blackout",
            "applicable": resistance_exists,
            "passed": not macro_risk.blocked,
            "detail": macro_risk,
        },
        {
            "step": "double_sweep_and_fomo_guard",
            "applicable": resistance_exists,
            "passed": not (trap.double_sweep or trap.fomo_extension),
            "detail": trap,
        },
        {
            "step": "15m_1h_4h_alignment",
            "applicable": resistance_exists and liquidity.entries_allowed,
            "passed": alignment_passed,
            "detail": {
                "candidate_side": candidate_side,
                "minimum_score": minimum_score,
                "scores": opportunity.momentum_scores,
            },
        },
        {
            "step": "15m_breakout_retest_or_rejection",
            "applicable": gate_direction_applicable,
            "passed": pattern_confirmed,
            "detail": {
                "long_retest_confirmed": gate.long_retest_confirmed,
                "short_rejection_confirmed": gate.short_rejection_confirmed,
            },
        },
        {
            "step": "1h_close_confirmation",
            "applicable": gate_direction_applicable,
            "passed": gate.hourly_confirmed,
            "detail": opportunity.hourly_structure,
        },
        {
            "step": "daily_context",
            "applicable": gate_direction_applicable,
            "passed": gate.daily_confirmed,
            "detail": opportunity.daily_structure,
        },
        {
            "step": "price_near_analysis_zone",
            "applicable": gate_direction_applicable,
            "passed": gate.near_resistance_confirmed,
            "detail": {
                "current_price": opportunity.peak_map.current_price,
                "resistance": gate.resistance,
            },
        },
        {
            "step": "execution_plan_viable",
            "applicable": execution_applicable,
            "passed": plan is not None,
            "detail": opportunity.execution_reason,
        },
        {
            "step": "position_size_viable",
            "applicable": sizing_applicable,
            "passed": opportunity.sized_plan is not None,
            "detail": opportunity.sized_plan,
        },
        {
            "step": "price_near_or_inside_entry",
            "applicable": proximity_applicable,
            "passed": phase is not None,
            "detail": {
                "phase": phase,
                "executable_price": executable_price,
                "distance_to_entry": entry_distance,
                "approach_buffer_pct": auto_settings.get("approach_buffer_pct", 0.01),
            },
        },
    ]
    blockers = [
        {
            "step": item["step"],
            "detail": item["detail"],
        }
        for item in decision_path
        if item["applicable"] and not item["passed"]
    ]

    recent_bars = diagnostic_settings.get(
        "recent_closed_candles",
        {"15min": 8, "1h": 6, "4h": 4, "1day": 3},
    )
    frames = {
        timeframe: _frame_snapshot(frame, int(recent_bars.get(timeframe, 3)))
        for timeframe, frame in opportunity.frames.items()
        if timeframe in ("15min", "1h", "4h", "1day")
    }
    analyzed_at = opportunity.analysis_now
    if analyzed_at.tzinfo is None:
        analyzed_at = analyzed_at.replace(tzinfo=timezone.utc)

    return _json_safe(
        {
            "schema_version": 1,
            "event": event,
            "analyzed_at_utc": analyzed_at.astimezone(timezone.utc),
            "analyzed_at_vn": analyzed_at.astimezone(VIETNAM_TIMEZONE),
            "symbol": config.get("symbol", "XAUUSDT"),
            "market": {
                "quote": opportunity.realtime_quote,
                "frames": frames,
            },
            "analysis_map": {
                "momentum": {
                    timeframe: {
                        "score": opportunity.momentum_scores.get(timeframe),
                        "label": bias.label,
                        "components": bias.components,
                    }
                    for timeframe, bias in opportunity.momentum_biases.items()
                },
                "hourly_structure": opportunity.hourly_structure,
                "daily_structure": opportunity.daily_structure,
                "peak_map": opportunity.peak_map,
                "liquidity": liquidity,
                "liquidity_trap": trap,
                "macro_risk": macro_risk,
            },
            "decision": {
                "candidate_side": candidate_side,
                "gate": gate,
                "execution_reason": opportunity.execution_reason,
                "execution_plan": plan,
                "sized_plan": opportunity.sized_plan,
                "setup_quality": quality,
                "alert_phase": phase,
                "executable_price": executable_price,
                "distance_to_entry": entry_distance,
                "path": decision_path,
                "blockers": blockers,
            },
            "thresholds": {
                "signal_confirmation": signal_settings,
                "peak_execution": execution_settings,
                "peak_liquidity": liquidity_settings,
                "liquidity_traps": config.get("liquidity_traps", {}),
                "macro_guard": config.get("macro_guard", {}),
                "setup_quality": config.get("setup_quality", {}),
                "auto_alerts": {
                    key: auto_settings.get(key)
                    for key in (
                        "poll_seconds",
                        "active_poll_seconds",
                        "notify_approaching",
                        "approach_buffer_pct",
                        "same_setup_cooldown_minutes",
                    )
                },
            },
        }
    )


def build_diagnostic_event(
    event: str,
    occurred_at: datetime,
    details: dict,
    symbol: str = "XAUUSDT",
) -> dict:
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    return _json_safe(
        {
            "schema_version": 1,
            "event": event,
            "analyzed_at_utc": occurred_at.astimezone(timezone.utc),
            "analyzed_at_vn": occurred_at.astimezone(VIETNAM_TIMEZONE),
            "symbol": symbol,
            "details": details,
        }
    )


def append_analysis_snapshot(record: dict, settings: dict) -> Path | None:
    if not settings.get("enabled", True):
        return None
    directory = Path(settings.get("directory", "logs/diagnostics"))
    directory.mkdir(parents=True, exist_ok=True)
    retention_days = max(1, int(settings.get("retention_days", 14)))
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    for old_file in directory.glob("analysis-*.jsonl"):
        try:
            modified = datetime.fromtimestamp(old_file.stat().st_mtime, tz=timezone.utc)
            if modified < cutoff:
                old_file.unlink()
        except OSError:
            continue

    analyzed_at_vn = datetime.fromisoformat(record["analyzed_at_vn"])
    date_label = analyzed_at_vn.strftime("%Y-%m-%d")
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    payload_size = len(payload.encode("utf-8"))
    max_bytes = max(1, int(settings.get("max_file_mb", 10))) * 1024 * 1024
    part = 1
    path = directory / f"analysis-{date_label}.jsonl"
    while path.exists() and path.stat().st_size + payload_size > max_bytes:
        part += 1
        path = directory / f"analysis-{date_label}-part{part:02d}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload)
    return path
