"""No-look-ahead replay of the live strategy on Exness Bid/Ask minute bars."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Mapping

import pandas as pd

from market_sessions import market_session
from position_exit import PositionState, assess_closed_candle_exit
from strategy_engine import (
    EMA200_WARMUP_BARS,
    MIN_BARS,
    TechnicalSnapshot,
    analyze_market,
    technical_frame,
)


INTERVAL_RULES = {
    "1min": ("1min", pd.Timedelta(minutes=1)),
    "5min": ("5min", pd.Timedelta(minutes=5)),
    "15min": ("15min", pd.Timedelta(minutes=15)),
    "1h": ("1h", pd.Timedelta(hours=1)),
    "4h": ("4h", pd.Timedelta(hours=4)),
    "1day": ("1D", pd.Timedelta(days=1)),
}


@dataclass(frozen=True)
class TradeRecord:
    variant: str
    symbol: str
    side: str
    setup: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    initial_stop: float
    invalidation_level: float
    realized_r: float
    exit_reason: str
    bars_held_5m: int
    tp1_hit: bool


def entry_is_allowed(
    previous_at: pd.Timestamp | None,
    previous_fingerprint: str,
    fingerprint: str,
    now: pd.Timestamp,
    cooldown_minutes: int,
) -> bool:
    if fingerprint == previous_fingerprint:
        return False
    return bool(
        previous_at is None
        or now - previous_at >= pd.Timedelta(minutes=max(0, cooldown_minutes))
    )


def build_timeframes(one_minute: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build UTC-aligned Bid OHLC frames whose index is candle open time."""
    source = one_minute[["open", "high", "low", "close", "volume"]].copy()
    frames: dict[str, pd.DataFrame] = {"1min": source}
    for interval, (rule, _) in INTERVAL_RULES.items():
        if interval == "1min":
            continue
        frame = source.resample(rule, label="left", closed="left", origin="start_day").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        frames[interval] = frame.dropna(subset=["open", "high", "low", "close"])
    return frames


def closed_frames_at(
    frames: Mapping[str, pd.DataFrame],
    closed_at: pd.Timestamp,
    tail: int = 260,
) -> dict[str, pd.DataFrame]:
    """Slice only candles that were fully closed at ``closed_at``."""
    result: dict[str, pd.DataFrame] = {}
    for interval, frame in frames.items():
        duration = INTERVAL_RULES[interval][1]
        cutoff = closed_at - duration
        result[interval] = frame.loc[:cutoff].tail(tail)
    return result


def _snapshot(interval: str, values: pd.Series) -> TechnicalSnapshot:
    ema200 = float(values["ema200"]) if pd.notna(values["ema200"]) else None
    return TechnicalSnapshot(
        interval=interval,
        close=float(values["close"]),
        rsi=float(values["rsi"]),
        rsi_slope=float(values["rsi_slope"]),
        ema20=float(values["ema20"]),
        ema50=float(values["ema50"]),
        ema200=ema200,
        macd_histogram=float(values["macd_histogram"]),
        adx=float(values["adx"]),
        atr=float(values["atr"]),
        score=float(values["score"]),
    )


def snapshots_at(
    frames: Mapping[str, pd.DataFrame],
    technicals: Mapping[str, pd.DataFrame],
    closed_at: pd.Timestamp,
    require_ema200: bool = True,
) -> dict[str, TechnicalSnapshot] | None:
    result: dict[str, TechnicalSnapshot] = {}
    for interval, frame in frames.items():
        cutoff = closed_at - INTERVAL_RULES[interval][1]
        position = frame.index.searchsorted(cutoff, side="right") - 1
        # Keep this at >= MIN_BARS: ta's ADX emits a literal 0.0 (not NaN) for
        # its first 27 bars, so the isna() check below cannot catch a cold ADX.
        if position < MIN_BARS - 1:
            return None
        values = technicals[interval].iloc[position]
        # EMA200 is only demanded from frames long enough to warm it up. A short
        # backtest has a few hundred daily bars, and technical_frame drops EMA200
        # there by design; requiring it anyway would reject every single bar. The
        # daily frame carries no weight in the entry decision regardless.
        interval_has_ema200 = require_ema200 and len(frame) >= EMA200_WARMUP_BARS
        required_values = values if interval_has_ema200 else values.drop(labels=["ema200"])
        if required_values.isna().any():
            return None
        result[interval] = _snapshot(interval, values)
    return result


def _price_r(position: PositionState, price: float) -> float:
    direction = 1.0 if position.side == "LONG" else -1.0
    return direction * (float(price) - position.entry_price) / position.initial_risk


def _record(
    variant: str,
    position: PositionState,
    when: pd.Timestamp,
    price: float,
    reason: str,
) -> TradeRecord:
    total_r = position.realized_r + position.remaining_fraction * _price_r(position, price)
    return TradeRecord(
        variant=variant,
        symbol=position.symbol,
        side=position.side,
        setup=position.setup,
        entry_time=position.entry_time,
        exit_time=when.to_pydatetime(),
        entry_price=position.entry_price,
        exit_price=float(price),
        initial_stop=position.initial_stop,
        invalidation_level=position.invalidation_level,
        realized_r=float(total_r),
        exit_reason=reason,
        bars_held_5m=position.bars_held_5m,
        tp1_hit=position.tp1_hit,
    )


def _worse_stop_fill(position: PositionState, bar: pd.Series) -> float:
    if position.side == "LONG":
        return min(position.current_stop, float(bar["bid_open"]))
    return max(position.current_stop, float(bar["ask_open"]))


def process_hard_levels(
    variant: str,
    position: PositionState,
    bar: pd.Series,
    closed_at: pd.Timestamp,
) -> TradeRecord | None:
    """Apply conservative one-minute Bid/Ask SL/TP ordering."""
    if position.side == "LONG":
        favorable = float(bar["bid_high"])
        stop_hit = float(bar["bid_low"]) <= position.current_stop
        tp1_hit = float(bar["bid_high"]) >= position.take_profit_1
        tp2_hit = float(bar["bid_high"]) >= position.take_profit_2
    else:
        favorable = float(bar["ask_low"])
        stop_hit = float(bar["ask_high"]) >= position.current_stop
        tp1_hit = float(bar["ask_low"]) <= position.take_profit_1
        tp2_hit = float(bar["ask_low"]) <= position.take_profit_2
    position.max_favorable_r = max(position.max_favorable_r, _price_r(position, favorable))

    # When both sides of a level are touched inside one minute, the exact tick
    # order is unknown after aggregation; stop-first is the conservative rule.
    if stop_hit:
        fill = _worse_stop_fill(position, bar)
        reason = "BREAK_EVEN_STOP" if position.tp1_hit else "HARD_STOP"
        return _record(variant, position, closed_at, fill, reason)
    if not position.tp1_hit and tp1_hit:
        closed_fraction = 0.5
        position.realized_r += closed_fraction * _price_r(position, position.take_profit_1)
        position.remaining_fraction -= closed_fraction
        position.tp1_hit = True
        position.current_stop = position.entry_price
        if tp2_hit:
            return _record(variant, position, closed_at, position.take_profit_2, "TP2_AFTER_TP1")
    elif position.tp1_hit and tp2_hit:
        return _record(variant, position, closed_at, position.take_profit_2, "TP2_AFTER_TP1")
    return None


def _new_position(symbol: str, asset_type: str, plan, when: pd.Timestamp) -> PositionState:
    return PositionState(
        symbol=symbol,
        asset_type=asset_type,
        side=plan.side,
        setup=plan.setup,
        entry_time=when.to_pydatetime(),
        entry_price=float(plan.preferred_entry),
        initial_stop=float(plan.stop_loss),
        current_stop=float(plan.stop_loss),
        take_profit_1=float(plan.take_profit_1),
        take_profit_2=float(plan.take_profit_2),
        invalidation_level=float(plan.invalidation_level),
        initial_risk=float(plan.risk),
    )


def summarize_trades(trades: list[TradeRecord], variant: str) -> dict:
    selected = [item for item in trades if item.variant == variant]
    values = pd.Series([item.realized_r for item in selected], dtype=float)
    if values.empty:
        return {
            "variant": variant,
            "trades": 0,
            "net_r": 0.0,
            "average_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "exit_reasons": {},
        }
    equity = pd.concat([pd.Series([0.0]), values.cumsum()], ignore_index=True)
    drawdown = equity.cummax() - equity
    gains = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    reasons = pd.Series(
        [item.exit_reason.split(":", 1)[0] for item in selected]
    ).value_counts().to_dict()
    return {
        "variant": variant,
        "trades": int(len(values)),
        "net_r": round(float(values.sum()), 3),
        "average_r": round(float(values.mean()), 3),
        "win_rate": round(float((values > 0).mean() * 100.0), 2),
        "profit_factor": round(gains / losses, 3) if losses > 0 else None,
        "max_drawdown_r": round(float(drawdown.max()), 3),
        "exit_reasons": {str(key): int(value) for key, value in reasons.items()},
    }


def replay_strategy(
    one_minute: pd.DataFrame,
    symbol: str,
    asset_type: str,
    strategy_settings: Mapping[str, object] | None = None,
    exit_settings: Mapping[str, object] | None = None,
    digits: int = 3,
    require_ema200: bool = True,
    entry_cooldown_minutes: int = 60,
    evaluation_start: str | datetime | None = None,
    evaluation_end: str | datetime | None = None,
) -> dict:
    """Replay baseline and managed-exit variants on identical Exness bars."""
    strategy_settings = dict(strategy_settings or {})
    exit_settings = dict(exit_settings or {})
    frames = build_timeframes(one_minute)
    technicals = {key: technical_frame(frame) for key, frame in frames.items()}
    active: dict[str, PositionState | None] = {"baseline": None, "managed_exit": None}
    last_entry_at: dict[str, pd.Timestamp | None] = {"baseline": None, "managed_exit": None}
    last_fingerprint: dict[str, str] = {"baseline": "", "managed_exit": ""}
    trades: list[TradeRecord] = []
    signal_count = 0

    replay_bars = one_minute
    if evaluation_start is not None:
        start = pd.Timestamp(evaluation_start)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        replay_bars = replay_bars.loc[start:]
    if evaluation_end is not None:
        end = pd.Timestamp(evaluation_end)
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        replay_bars = replay_bars.loc[:end]

    for opened_at, bar in replay_bars.iterrows():
        closed_at = opened_at + pd.Timedelta(minutes=1)
        hard_exited_now: set[str] = set()
        for variant, position in list(active.items()):
            if position is None:
                continue
            record = process_hard_levels(variant, position, bar, closed_at)
            if record is not None:
                trades.append(record)
                active[variant] = None
                hard_exited_now.add(variant)

        if closed_at.minute % 5 != 0:
            continue
        closed_frames = closed_frames_at(frames, closed_at)
        snapshots = snapshots_at(
            frames,
            technicals,
            closed_at,
            require_ema200=require_ema200,
        )
        if snapshots is None:
            continue

        exited_now: set[str] = set(hard_exited_now)
        for variant, position in list(active.items()):
            if position is None:
                continue
            position.bars_held_5m += 1
            if variant != "managed_exit":
                continue
            decision = assess_closed_candle_exit(position, closed_frames, snapshots, exit_settings)
            if decision.action == "MOVE_STOP" and decision.new_stop is not None:
                position.current_stop = float(decision.new_stop)
            elif decision.action == "FULL_EXIT":
                price = float(bar["bid_close"] if position.side == "LONG" else bar["ask_close"])
                trades.append(_record(variant, position, closed_at, price, decision.reason))
                active[variant] = None
                exited_now.add(variant)

        session = market_session(asset_type, closed_at.to_pydatetime())
        if not session.is_open:
            continue
        quote = {
            "bid": float(bar["bid_close"]),
            "ask": float(bar["ask_close"]),
            "close": float(bar["bid_close"]),
            "digits": digits,
        }
        analysis = analyze_market(
            symbol,
            asset_type,
            closed_frames,
            quote,
            session,
            strategy_settings,
            closed_at.to_pydatetime(),
            precomputed_snapshots=snapshots,
        )
        if analysis.plan is None:
            continue
        signal_count += 1
        fingerprint = "|".join(
            (
                analysis.plan.side,
                analysis.plan.setup,
                f"{analysis.plan.preferred_entry:.8g}",
                f"{analysis.plan.stop_loss:.8g}",
            )
        )
        for variant in active:
            if (
                active[variant] is None
                and variant not in exited_now
                and entry_is_allowed(
                    last_entry_at[variant],
                    last_fingerprint[variant],
                    fingerprint,
                    closed_at,
                    entry_cooldown_minutes,
                )
            ):
                active[variant] = _new_position(symbol, asset_type, analysis.plan, closed_at)
                last_entry_at[variant] = closed_at
                last_fingerprint[variant] = fingerprint

    if not replay_bars.empty:
        last_at = replay_bars.index[-1] + pd.Timedelta(minutes=1)
        last = replay_bars.iloc[-1]
        for variant, position in active.items():
            if position is not None:
                price = float(last["bid_close"] if position.side == "LONG" else last["ask_close"])
                trades.append(_record(variant, position, last_at, price, "END_OF_DATA"))

    return {
        "source_bars_1m": int(len(replay_bars)),
        "start": replay_bars.index.min().isoformat() if not replay_bars.empty else None,
        "end": replay_bars.index.max().isoformat() if not replay_bars.empty else None,
        "signal_events": signal_count,
        "summaries": {
            variant: summarize_trades(trades, variant)
            for variant in ("baseline", "managed_exit")
        },
        "trades": [asdict(item) for item in trades],
    }
