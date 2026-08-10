"""Deterministic 20:30 opening-range breakout/retest strategy for XAUUSDT."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from math import ceil, floor
from zoneinfo import ZoneInfo

import pandas as pd


VIETNAM_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
M5_INTERVAL = timedelta(minutes=5)


@dataclass(frozen=True)
class M5SessionWindow:
    session_date: date
    reference_start: datetime
    monitor_start: datetime
    monitor_end: datetime

    @property
    def active_from(self) -> datetime:
        return self.monitor_start


@dataclass(frozen=True)
class OpeningRangeM5Assessment:
    status: str
    reason: str
    checked_at: datetime
    window: M5SessionWindow
    opening_high: float | None = None
    opening_low: float | None = None
    breakout_side: str | None = None
    breakout_candle_start: datetime | None = None
    breakout_close: float | None = None
    confirmation_candle_start: datetime | None = None
    confirmation_close: float | None = None
    structural_stop: float | None = None

    @property
    def confirmation_closed_at(self) -> datetime | None:
        if self.confirmation_candle_start is None:
            return None
        return self.confirmation_candle_start + M5_INTERVAL


@dataclass(frozen=True)
class M5TradePlan:
    side: str
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk: float
    reward_risk_1: float
    reward_risk_2: float


def session_window(now: datetime) -> M5SessionWindow:
    """Return today's fixed Vietnam-time M5 opening-range window."""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(VIETNAM_TZ)
    session_date = local_now.date()
    reference_start = datetime.combine(
        session_date,
        time(20, 30),
        tzinfo=VIETNAM_TZ,
    )
    monitor_start = reference_start + M5_INTERVAL
    monitor_end = datetime.combine(
        session_date + timedelta(days=1),
        time.min,
        tzinfo=VIETNAM_TZ,
    )
    return M5SessionWindow(
        session_date=session_date,
        reference_start=reference_start,
        monitor_start=monitor_start,
        monitor_end=monitor_end,
    )


def is_monitoring_time(now: datetime) -> bool:
    window = session_window(now)
    local_now = _aware_utc(now).astimezone(VIETNAM_TZ)
    return window.monitor_start <= local_now < window.monitor_end


def seconds_until_next_m5_check(
    now: datetime,
    settle_seconds: int = 2,
) -> float:
    """Delay until just after the next wall-clock M5 close boundary."""
    now_utc = _aware_utc(now)
    settle_seconds = max(0, min(30, int(settle_seconds)))
    boundary = now_utc.replace(second=0, microsecond=0)
    boundary += timedelta(minutes=(-now_utc.minute) % 5)
    candidate = boundary + timedelta(seconds=settle_seconds)
    if candidate <= now_utc:
        candidate += M5_INTERVAL
    return (candidate - now_utc).total_seconds()


def assess_opening_range_m5(
    frame: pd.DataFrame,
    now: datetime,
    settings: dict | None = None,
) -> OpeningRangeM5Assessment:
    """Assess the session using only M5 candles that had closed by ``now``."""
    settings = settings or {}
    checked_at = _aware_utc(now)
    window = session_window(checked_at)
    local_now = checked_at.astimezone(VIETNAM_TZ)
    if not window.monitor_start <= local_now < window.monitor_end:
        return OpeningRangeM5Assessment(
            status="INACTIVE",
            reason="Ngoài khung theo dõi 20:35–24:00 giờ Việt Nam.",
            checked_at=checked_at,
            window=window,
        )

    required_columns = {"open", "high", "low", "close"}
    missing = required_columns.difference(frame.columns)
    if missing:
        return OpeningRangeM5Assessment(
            status="DATA_ERROR",
            reason="Thiếu cột nến M5: " + ", ".join(sorted(missing)) + ".",
            checked_at=checked_at,
            window=window,
        )

    closed = _closed_m5_frame(frame, checked_at)
    reference_start_utc = window.reference_start.astimezone(timezone.utc)
    reference_rows = closed.loc[
        closed.index == pd.Timestamp(reference_start_utc)
    ]
    if reference_rows.empty:
        return OpeningRangeM5Assessment(
            status="WAIT_REFERENCE",
            reason="Chưa nhận được nến M5 20:30–20:35 đã đóng.",
            checked_at=checked_at,
            window=window,
        )

    reference = reference_rows.iloc[-1]
    opening_high = float(reference["high"])
    opening_low = float(reference["low"])
    if not opening_high > opening_low:
        return OpeningRangeM5Assessment(
            status="DATA_ERROR",
            reason="Nến 20:30–20:35 có High/Low không hợp lệ.",
            checked_at=checked_at,
            window=window,
            opening_high=opening_high,
            opening_low=opening_low,
        )

    monitor_start_utc = window.monitor_start.astimezone(timezone.utc)
    monitor_end_utc = window.monitor_end.astimezone(timezone.utc)
    candidates = closed.loc[
        (closed.index >= pd.Timestamp(monitor_start_utc))
        & (closed.index < pd.Timestamp(monitor_end_utc))
    ]
    breakout_buffer = max(
        0.0,
        float(settings.get("breakout_buffer_price", 0.0)),
    )
    retest_tolerance = max(
        0.0,
        float(settings.get("retest_tolerance_price", 0.50)),
    )
    minimum_close_location = min(
        1.0,
        max(0.5, float(settings.get("minimum_rejection_close_location", 0.65))),
    )
    stop_buffer = max(
        0.0,
        float(settings.get("stop_buffer_price", 0.20)),
    )

    pending_side: str | None = None
    breakout_start: datetime | None = None
    breakout_close: float | None = None

    for timestamp, candle in candidates.iterrows():
        candle_start = timestamp.to_pydatetime().astimezone(timezone.utc)
        candle_open = float(candle["open"])
        candle_high = float(candle["high"])
        candle_low = float(candle["low"])
        candle_close = float(candle["close"])

        if pending_side is None:
            if candle_close > opening_high + breakout_buffer:
                pending_side = "LONG"
                breakout_start = candle_start
                breakout_close = candle_close
            elif candle_close < opening_low - breakout_buffer:
                pending_side = "SHORT"
                breakout_start = candle_start
                breakout_close = candle_close
            # A breakout candle can never also be its own retest confirmation.
            continue

        if pending_side == "LONG":
            if candle_close < opening_low - breakout_buffer:
                pending_side = "SHORT"
                breakout_start = candle_start
                breakout_close = candle_close
                continue
            if candle_close <= opening_high:
                pending_side = None
                breakout_start = None
                breakout_close = None
                continue
            if _is_long_retest_confirmation(
                candle_open,
                candle_high,
                candle_low,
                candle_close,
                opening_high,
                retest_tolerance,
                minimum_close_location,
            ):
                return OpeningRangeM5Assessment(
                    status="SIGNAL",
                    reason="Breakout LONG đã có nến M5 sau đó retest và từ chối vùng High.",
                    checked_at=checked_at,
                    window=window,
                    opening_high=opening_high,
                    opening_low=opening_low,
                    breakout_side="LONG",
                    breakout_candle_start=breakout_start,
                    breakout_close=breakout_close,
                    confirmation_candle_start=candle_start,
                    confirmation_close=candle_close,
                    structural_stop=candle_low - stop_buffer,
                )
        else:
            if candle_close > opening_high + breakout_buffer:
                pending_side = "LONG"
                breakout_start = candle_start
                breakout_close = candle_close
                continue
            if candle_close >= opening_low:
                pending_side = None
                breakout_start = None
                breakout_close = None
                continue
            if _is_short_retest_confirmation(
                candle_open,
                candle_high,
                candle_low,
                candle_close,
                opening_low,
                retest_tolerance,
                minimum_close_location,
            ):
                return OpeningRangeM5Assessment(
                    status="SIGNAL",
                    reason="Breakout SHORT đã có nến M5 sau đó retest và từ chối vùng Low.",
                    checked_at=checked_at,
                    window=window,
                    opening_high=opening_high,
                    opening_low=opening_low,
                    breakout_side="SHORT",
                    breakout_candle_start=breakout_start,
                    breakout_close=breakout_close,
                    confirmation_candle_start=candle_start,
                    confirmation_close=candle_close,
                    structural_stop=candle_high + stop_buffer,
                )

    if pending_side is not None:
        boundary_name = "High" if pending_side == "LONG" else "Low"
        return OpeningRangeM5Assessment(
            status="WAIT_RETEST",
            reason=f"Đã breakout {pending_side}; đang chờ nến M5 sau retest vùng {boundary_name}.",
            checked_at=checked_at,
            window=window,
            opening_high=opening_high,
            opening_low=opening_low,
            breakout_side=pending_side,
            breakout_candle_start=breakout_start,
            breakout_close=breakout_close,
        )

    return OpeningRangeM5Assessment(
        status="WAIT_BREAKOUT",
        reason="Đã khóa biên 20:30–20:35; chưa có nến M5 đóng phá High/Low hợp lệ.",
        checked_at=checked_at,
        window=window,
        opening_high=opening_high,
        opening_low=opening_low,
    )


def build_m5_trade_plan(
    assessment: OpeningRangeM5Assessment,
    executable_price: float,
    settings: dict | None = None,
) -> M5TradePlan:
    """Build exact 1.5R/2R targets from the executable price and structural SL."""
    settings = settings or {}
    if (
        assessment.status != "SIGNAL"
        or assessment.breakout_side not in {"LONG", "SHORT"}
        or assessment.structural_stop is None
    ):
        raise ValueError("M5 trade plan requires a confirmed SIGNAL assessment")

    price_tick = max(1e-9, float(settings.get("price_tick", 0.01)))
    entry = _round_nearest(float(executable_price), price_tick)
    if assessment.breakout_side == "LONG":
        stop_loss = _round_down(float(assessment.structural_stop), price_tick)
        risk = entry - stop_loss
        direction = 1.0
    else:
        stop_loss = _round_up(float(assessment.structural_stop), price_tick)
        risk = stop_loss - entry
        direction = -1.0
    if risk <= 0:
        raise ValueError("Executable price is already beyond the structural stop")

    reward_risk_1 = max(1.0, float(settings.get("take_profit_1_r", 1.5)))
    reward_risk_2 = max(
        reward_risk_1,
        float(settings.get("take_profit_2_r", 2.0)),
    )
    take_profit_1 = _round_nearest(
        entry + direction * risk * reward_risk_1,
        price_tick,
    )
    take_profit_2 = _round_nearest(
        entry + direction * risk * reward_risk_2,
        price_tick,
    )
    return M5TradePlan(
        side=assessment.breakout_side,
        entry=entry,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        risk=risk,
        reward_risk_1=reward_risk_1,
        reward_risk_2=reward_risk_2,
    )


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _closed_m5_frame(frame: pd.DataFrame, now: datetime) -> pd.DataFrame:
    closed = frame.copy()
    index = pd.DatetimeIndex(closed.index)
    if index.tz is None:
        index = index.tz_localize(timezone.utc)
    else:
        index = index.tz_convert(timezone.utc)
    closed.index = index
    closed = closed.sort_index()
    return closed.loc[closed.index + M5_INTERVAL <= pd.Timestamp(now)]


def _is_long_retest_confirmation(
    candle_open: float,
    candle_high: float,
    candle_low: float,
    candle_close: float,
    boundary: float,
    tolerance: float,
    minimum_close_location: float,
) -> bool:
    candle_range = candle_high - candle_low
    if candle_range <= 0:
        return False
    touched_zone = candle_low <= boundary + tolerance and candle_high >= boundary - tolerance
    close_location = (candle_close - candle_low) / candle_range
    return (
        touched_zone
        and candle_close > boundary
        and candle_close > candle_open
        and close_location >= minimum_close_location
    )


def _is_short_retest_confirmation(
    candle_open: float,
    candle_high: float,
    candle_low: float,
    candle_close: float,
    boundary: float,
    tolerance: float,
    minimum_close_location: float,
) -> bool:
    candle_range = candle_high - candle_low
    if candle_range <= 0:
        return False
    touched_zone = candle_high >= boundary - tolerance and candle_low <= boundary + tolerance
    close_location = (candle_close - candle_low) / candle_range
    return (
        touched_zone
        and candle_close < boundary
        and candle_close < candle_open
        and close_location <= 1.0 - minimum_close_location
    )


def _round_nearest(value: float, tick: float) -> float:
    return round(value / tick) * tick


def _round_down(value: float, tick: float) -> float:
    return floor((value + 1e-12) / tick) * tick


def _round_up(value: float, tick: float) -> float:
    return ceil((value - 1e-12) / tick) * tick
