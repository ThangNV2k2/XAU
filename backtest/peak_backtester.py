from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math

import pandas as pd

from indicators.signal_engine import compute_momentum_bias_series
from macro_risk import assess_macro_risk
from market_context import find_chart_structure, interval_duration
from peak_analysis import (
    assess_liquidity_traps,
    assess_peak_liquidity,
    assess_peak_trade_gate,
    assess_setup_quality,
    build_peak_execution_plan,
    build_peak_map,
)


@dataclass(frozen=True)
class PeakBacktestTrade:
    side: str
    tier: str
    score: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    stop: float
    tp1: float
    tp2: float
    gross_r: float
    cost_r: float
    net_r: float
    exit_reason: str


@dataclass
class _Position:
    side: str
    tier: str
    score: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry: float
    stop: float
    tp1: float
    tp2: float
    risk: float
    cost_r: float
    maximum_holding_bars: int
    bars_held: int = 0
    tp1_hit: bool = False


@dataclass
class _Pending:
    side: str
    tier: str
    score: int
    signal_time: pd.Timestamp
    expires_at: pd.Timestamp
    entry: float
    stop: float
    tp1: float
    tp2: float
    cost_r: float
    maximum_holding_bars: int


def _frames_at(
    frames: dict[str, pd.DataFrame],
    analysis_now: pd.Timestamp,
    sizes: dict,
) -> dict[str, pd.DataFrame]:
    selected = {}
    for timeframe, frame in frames.items():
        duration = interval_duration(timeframe)
        last_open = analysis_now - duration
        end_position = int(frame.index.searchsorted(last_open, side="right"))
        size = max(40, min(1500, int(sizes.get(timeframe, 200))))
        selected[timeframe] = frame.iloc[max(0, end_position - size) : end_position]
    return selected


def _close_trade(
    position: _Position,
    exit_time: pd.Timestamp,
    gross_r: float,
    reason: str,
) -> PeakBacktestTrade:
    return PeakBacktestTrade(
        side=position.side,
        tier=position.tier,
        score=position.score,
        signal_time=position.signal_time,
        entry_time=position.entry_time,
        exit_time=exit_time,
        entry=position.entry,
        stop=position.stop,
        tp1=position.tp1,
        tp2=position.tp2,
        gross_r=gross_r,
        cost_r=position.cost_r,
        net_r=gross_r - position.cost_r,
        exit_reason=reason,
    )


def _advance_position(
    position: _Position,
    timestamp: pd.Timestamp,
    row: pd.Series,
) -> tuple[_Position | None, PeakBacktestTrade | None]:
    position.bars_held += 1
    high, low, close = float(row.high), float(row.low), float(row.close)
    is_long = position.side == "LONG"
    stop_level = position.entry if position.tp1_hit else position.stop
    stop_hit = low <= stop_level if is_long else high >= stop_level
    tp2_hit = high >= position.tp2 if is_long else low <= position.tp2
    tp1_hit = high >= position.tp1 if is_long else low <= position.tp1

    # Conservative OHLC assumption: if stop and target are both touched inside a
    # candle, count the stop first because the intrabar path is unknown.
    if stop_hit:
        gross_r = 0.5 if position.tp1_hit else -1.0
        reason = "BREAKEVEN_AFTER_TP1" if position.tp1_hit else "STOP"
        return None, _close_trade(position, timestamp, gross_r, reason)
    if tp2_hit:
        rr2 = abs(position.tp2 - position.entry) / position.risk
        gross_r = 0.5 + 0.5 * rr2
        return None, _close_trade(position, timestamp, gross_r, "TP2")
    if tp1_hit and not position.tp1_hit:
        position.tp1_hit = True
    if position.bars_held >= position.maximum_holding_bars:
        direction = 1 if is_long else -1
        open_r = direction * (close - position.entry) / position.risk
        gross_r = 0.5 + 0.5 * open_r if position.tp1_hit else open_r
        return None, _close_trade(position, timestamp, gross_r, "TIME_STOP")
    return position, None


def _wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Return a 95% Wilson interval without pretending tiny samples are precise."""
    if total <= 0:
        return None
    proportion = wins / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return (
        max(0.0, (centre - margin) / denominator),
        min(1.0, (centre + margin) / denominator),
    )


def _trade_statistics(trades: list[PeakBacktestTrade]) -> dict:
    net_r = pd.Series([trade.net_r for trade in trades], dtype=float)
    wins = net_r[net_r > 0]
    losses = net_r[net_r < 0]
    interval = _wilson_interval(len(wins), len(net_r))
    return {
        "trades": len(trades),
        "wins": len(wins),
        "win_rate_pct": round(float(len(wins) / len(net_r) * 100), 2)
        if len(net_r)
        else 0.0,
        "win_rate_95pct_wilson": (
            [round(interval[0] * 100, 2), round(interval[1] * 100, 2)]
            if interval is not None
            else None
        ),
        "profit_factor": (
            round(float(wins.sum() / abs(losses.sum())), 2)
            if len(losses) and abs(losses.sum()) > 0
            else None
        ),
        "average_net_r": round(float(net_r.mean()), 3) if len(net_r) else 0.0,
        "total_net_r": round(float(net_r.sum()), 3) if len(net_r) else 0.0,
    }


def run_peak_backtest(
    frames: dict[str, pd.DataFrame],
    config: dict,
) -> tuple[list[PeakBacktestTrade], dict]:
    base = frames["15min"].sort_index()
    peak_settings = config.get("peak_map", {})
    sizes = peak_settings.get("outputsize_by_timeframe", {})
    execution_settings = {
        **config.get("trading_plan", {}),
        **config.get("peak_execution", {}),
    }
    fee_pct = max(
        0.0,
        float(config.get("trading_plan", {}).get("estimated_round_trip_fee_pct", 0.10)),
    )
    slippage_bps = max(
        0.0,
        float(config.get("trading_plan", {}).get("estimated_slippage_bps", 2.0)),
    )
    timeout_bars = max(
        1,
        int(config.get("auto_alerts", {}).get("setup_timeout_minutes", 90)) // 15,
    )
    moderate_holding_bars = max(
        1,
        int(config.get("trading_plan", {}).get("moderate_bias_holding_bars", 8)),
    )
    strong_holding_bars = max(
        1,
        int(config.get("trading_plan", {}).get("strong_bias_holding_bars", 16)),
    )
    assumed_spread_bps = max(
        0.0,
        float(config.get("backtest", {}).get("assumed_spread_bps", 1.0)),
    )

    trades: list[PeakBacktestTrade] = []
    pending: _Pending | None = None
    position: _Position | None = None
    evaluated = 0
    eligible_setups = 0
    minimum_daily_bars = max(40, int(config.get("backtest", {}).get("minimum_daily_bars", 40)))
    bias_frames = {
        timeframe: compute_momentum_bias_series(frame, config["weights"])
        for timeframe, frame in frames.items()
    }
    minimum_score = float(
        config.get("signal_confirmation", {}).get("minimum_timeframe_score", 0.12)
    )
    evidence_cache: dict = {}

    for timestamp, row in base.iterrows():
        analysis_now = timestamp + timedelta(minutes=15)

        if position is not None:
            position, closed_trade = _advance_position(
                position,
                timestamp,
                row,
            )
            if closed_trade is not None:
                trades.append(closed_trade)
            continue

        if pending is not None:
            high, low = float(row.high), float(row.low)
            touched = low <= pending.entry <= high
            expired = analysis_now > pending.expires_at
            invalidated = (
                low <= pending.stop
                if pending.side == "LONG"
                else high >= pending.stop
            )
            if touched:
                risk = abs(pending.entry - pending.stop)
                position = _Position(
                    side=pending.side,
                    tier=pending.tier,
                    score=pending.score,
                    signal_time=pending.signal_time,
                    entry_time=timestamp,
                    entry=pending.entry,
                    stop=pending.stop,
                    tp1=pending.tp1,
                    tp2=pending.tp2,
                    risk=risk,
                    cost_r=pending.cost_r,
                    maximum_holding_bars=pending.maximum_holding_bars,
                )
                pending = None
                position, closed_trade = _advance_position(
                    position,
                    timestamp,
                    row,
                )
                if closed_trade is not None:
                    trades.append(closed_trade)
            elif invalidated or expired:
                pending = None
            continue

        point_frames = _frames_at(frames, analysis_now, sizes)
        if any(len(frame) < 40 for frame in point_frames.values()):
            continue
        if len(point_frames["1day"]) < minimum_daily_bars:
            continue
        evaluated += 1
        scores = {
            timeframe: float(
                bias_frames[timeframe].loc[point_frame.index[-1], "composite"]
            )
            for timeframe, point_frame in point_frames.items()
        }
        required_scores = [scores[timeframe] for timeframe in ("15min", "1h", "4h")]
        if not (
            all(score >= minimum_score for score in required_scores)
            or all(score <= -minimum_score for score in required_scores)
        ):
            continue
        current_price = float(point_frames["15min"].close.iloc[-1])
        peak_window = max(40, int(peak_settings.get("outputsize", 200)))
        peak_map = build_peak_map(
            {
                timeframe: frame.tail(peak_window)
                for timeframe, frame in point_frames.items()
            },
            current_price,
            peak_settings,
            evidence_cache=evidence_cache,
        )
        half_spread = current_price * assumed_spread_bps / 20_000
        liquidity = assess_peak_liquidity(
            point_frames["1h"],
            current_price,
            current_price - half_spread,
            current_price + half_spread,
            analysis_now=analysis_now.to_pydatetime(),
            settings=config.get("peak_liquidity", {}),
        )
        trap = assess_liquidity_traps(
            peak_map,
            point_frames["15min"],
            config.get("liquidity_traps", {}),
        )
        macro = assess_macro_risk(
            analysis_now.to_pydatetime(),
            config.get("macro_guard", {}),
        )
        daily_structure = find_chart_structure(point_frames["1day"])
        gate = assess_peak_trade_gate(
            peak_map,
            point_frames["15min"],
            scores,
            {
                **peak_settings,
                **config.get("signal_confirmation", {}),
                **config.get("peak_execution", {}),
            },
            frame_1h=point_frames["1h"],
            liquidity=liquidity,
            daily_pattern=daily_structure.pattern,
            trap=trap,
            macro_risk=macro,
        )
        plan, _ = build_peak_execution_plan(
            peak_map,
            gate,
            point_frames["15min"],
            execution_settings,
        )
        quality = assess_setup_quality(
            gate,
            plan,
            liquidity,
            trap,
            macro,
            {
                **config.get("setup_quality", {}),
                "base_risk_per_trade_pct": config.get("trading_plan", {}).get(
                    "risk_per_trade_pct", 0.5
                ),
            },
        )
        if plan is None or not quality.actionable:
            continue
        eligible_setups += 1
        risk = abs(plan.entry_reference - plan.stop_loss)
        if risk <= 0:
            continue
        estimated_cost = (
            plan.entry_reference * fee_pct / 100
            + plan.entry_reference * slippage_bps / 10_000
        )
        pending = _Pending(
            side=plan.side,
            tier=quality.tier,
            score=quality.score,
            signal_time=timestamp,
            expires_at=analysis_now + timedelta(minutes=timeout_bars * 15),
            entry=plan.entry_reference,
            stop=plan.stop_loss,
            tp1=plan.take_profit_1,
            tp2=plan.take_profit_2,
            cost_r=estimated_cost / risk,
            maximum_holding_bars=(
                strong_holding_bars
                if abs(scores["15min"]) >= 0.5
                else moderate_holding_bars
            ),
        )

    net_r = pd.Series([trade.net_r for trade in trades], dtype=float)
    equity = pd.concat([pd.Series([0.0]), net_r], ignore_index=True).cumsum()
    drawdown = equity - equity.cummax()
    overall = _trade_statistics(trades)
    by_tier = {
        tier: _trade_statistics([trade for trade in trades if trade.tier == tier])
        for tier in sorted({trade.tier for trade in trades})
    }
    validation = config.get("backtest", {}).get("validation", {})
    minimum_validation_trades = max(1, int(validation.get("minimum_trades", 100)))
    minimum_profit_factor = float(validation.get("minimum_profit_factor", 1.20))
    minimum_average_net_r = float(validation.get("minimum_average_net_r", 0.0))
    validation_passed = (
        len(trades) >= minimum_validation_trades
        and overall["profit_factor"] is not None
        and overall["profit_factor"] >= minimum_profit_factor
        and overall["average_net_r"] > minimum_average_net_r
    )
    stats = {
        "evaluated_closed_15m_bars": evaluated,
        "eligible_setup_snapshots": eligible_setups,
        "filled_trades": overall.pop("trades"),
        **overall,
        "max_drawdown_r": round(float(drawdown.min()), 3),
        "by_tier": by_tier,
        "validation": {
            "passed": validation_passed,
            "minimum_trades": minimum_validation_trades,
            "minimum_profit_factor": minimum_profit_factor,
            "minimum_average_net_r": minimum_average_net_r,
            "status": (
                "PASSED"
                if validation_passed
                else "INSUFFICIENT_OR_NO_EDGE_KEEP_PAPER_ONLY"
            ),
            "score_is_win_probability": False,
        },
        "assumptions": {
            "signal_execution": "limit eligible from next 15m bar",
            "same_bar_stop_target": "stop first (conservative)",
            "tp1": "close 50%, move remaining stop to entry",
            "round_trip_fee_pct": fee_pct,
            "slippage_bps": slippage_bps,
            "spread_bps": assumed_spread_bps,
        },
    }
    return trades, stats
