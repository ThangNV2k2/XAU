from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange


TIMEFRAME_LABELS = {
    "15min": "15m",
    "1h": "1H",
    "4h": "4H",
    "1day": "D1",
}
TIMEFRAME_WEIGHTS = {
    "15min": 1,
    "1h": 3,
    "4h": 5,
    "1day": 7,
}
VIETNAM_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")


def is_weekend_entry_blackout(
    analysis_now: datetime,
    cutoff_hour_vn: int = 5,
) -> bool:
    """Block new entries from Saturday cutoff through the end of Sunday in Vietnam."""
    if analysis_now.tzinfo is None:
        analysis_now = analysis_now.replace(tzinfo=timezone.utc)
    local_now = analysis_now.astimezone(VIETNAM_TIMEZONE)
    cutoff_hour = max(0, min(23, int(cutoff_hour_vn)))
    return local_now.weekday() == 6 or (
        local_now.weekday() == 5 and local_now.hour >= cutoff_hour
    )


@dataclass(frozen=True)
class FractalPivot:
    timeframe: str
    timestamp: pd.Timestamp
    price: float
    direction: str
    bar_position: int


@dataclass(frozen=True)
class PeakEvidence:
    timeframe: str
    timestamp: pd.Timestamp
    price: float
    age_bars: int
    zigzag_confirmed: bool
    reaction_atr: float
    volume_ratio: float | None


@dataclass
class PeakZone:
    lower: float
    upper: float
    center: float
    reliability: str
    score: int
    age_label: str
    newest_at: pd.Timestamp
    timeframes: tuple[str, ...]
    evidence_count: int
    zigzag_count: int
    reaction_atr: float
    volume_spike: bool
    status: str
    distance: float
    support_confirmed: bool


@dataclass
class PeakMap:
    current_price: float
    resistance_zones: list[PeakZone]
    converted_support_zones: list[PeakZone]
    volume_available: bool
    scanned_peak_count: int


@dataclass
class PeakTradeGate:
    allowed_decision: str
    reason: str
    resistance: PeakZone | None
    support: PeakZone | None
    long_retest_confirmed: bool
    short_rejection_confirmed: bool
    multi_timeframe_aligned: bool
    hourly_confirmed: bool = False
    liquidity_confirmed: bool = True
    daily_confirmed: bool = False


@dataclass
class PeakExecutionPlan:
    side: str
    entry_lower: float
    entry_upper: float
    entry_reference: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    reward_risk_1: float
    reward_risk_2: float
    structural_target: str


@dataclass
class PeakLiquidityAssessment:
    is_weekend: bool
    status: str
    recent_hourly_volume: float | None
    weekday_baseline_volume: float | None
    volume_ratio: float | None
    spread_bps: float | None
    entries_allowed: bool
    reason: str


def find_confirmed_fractals(
    df: pd.DataFrame,
    timeframe: str,
    span: int = 2,
) -> list[FractalPivot]:
    """Find confirmed pivot highs/lows using closed bars on both sides."""
    if span < 1 or len(df) < span * 2 + 1:
        return []

    highs = pd.to_numeric(df["high"], errors="coerce")
    lows = pd.to_numeric(df["low"], errors="coerce")
    pivots: list[FractalPivot] = []
    for position in range(span, len(df) - span):
        high = float(highs.iloc[position])
        low = float(lows.iloc[position])
        left_highs = highs.iloc[position - span : position]
        right_highs = highs.iloc[position + 1 : position + span + 1]
        left_lows = lows.iloc[position - span : position]
        right_lows = lows.iloc[position + 1 : position + span + 1]

        if high > float(left_highs.max()) and high >= float(right_highs.max()):
            pivots.append(
                FractalPivot(
                    timeframe=timeframe,
                    timestamp=df.index[position],
                    price=high,
                    direction="high",
                    bar_position=position,
                )
            )
        if low < float(left_lows.min()) and low <= float(right_lows.min()):
            pivots.append(
                FractalPivot(
                    timeframe=timeframe,
                    timestamp=df.index[position],
                    price=low,
                    direction="low",
                    bar_position=position,
                )
            )
    return sorted(pivots, key=lambda pivot: (pivot.timestamp, pivot.direction))


def filter_zigzag_pivots(
    pivots: list[FractalPivot],
    reversal_pct: float,
) -> list[FractalPivot]:
    """Keep alternating pivots whose reversal reaches the configured percentage."""
    if not pivots:
        return []
    threshold = max(0.0, float(reversal_pct)) / 100
    selected: list[FractalPivot] = []
    for pivot in pivots:
        if not selected:
            selected.append(pivot)
            continue
        previous = selected[-1]
        if pivot.direction == previous.direction:
            more_extreme = (
                pivot.price > previous.price
                if pivot.direction == "high"
                else pivot.price < previous.price
            )
            if more_extreme:
                selected[-1] = pivot
            continue

        reversal = abs(pivot.price - previous.price) / max(abs(previous.price), 1e-9)
        if reversal >= threshold:
            selected.append(pivot)
    return selected


def _atr_series(df: pd.DataFrame) -> pd.Series:
    if len(df) < 15:
        fallback = (df["high"] - df["low"]).rolling(5, min_periods=1).mean()
        return fallback.clip(lower=1e-9)
    return AverageTrueRange(
        pd.to_numeric(df["high"], errors="coerce"),
        pd.to_numeric(df["low"], errors="coerce"),
        pd.to_numeric(df["close"], errors="coerce"),
        window=14,
    ).average_true_range().replace(0, pd.NA).ffill().bfill().fillna(1e-9)


def _volume_ratio_at(df: pd.DataFrame, position: int) -> float | None:
    if "volume" not in df.columns or position < 1:
        return None
    volume = pd.to_numeric(df["volume"], errors="coerce")
    start = max(0, position - 20)
    baseline = float(volume.iloc[start:position].mean())
    current = float(volume.iloc[position])
    if pd.isna(baseline) or pd.isna(current) or baseline <= 0:
        return None
    return current / baseline


def _build_peak_evidence(
    frames: dict[str, pd.DataFrame],
    settings: dict,
) -> tuple[list[PeakEvidence], bool]:
    span = max(1, int(settings.get("fractal_span", 2)))
    reversal_settings = settings.get("zigzag_reversal_pct", {})
    reaction_bars = max(3, int(settings.get("reaction_lookahead_bars", 20)))
    evidence: list[PeakEvidence] = []
    volume_available = False

    for timeframe, df in frames.items():
        fractals = find_confirmed_fractals(df, timeframe, span)
        reversal_pct = float(reversal_settings.get(timeframe, 0.25))
        zigzag = filter_zigzag_pivots(fractals, reversal_pct)
        # The final ZigZag pivot can still move until an opposite swing is confirmed.
        confirmed_zigzag_highs = {
            pivot.timestamp
            for pivot in zigzag[:-1]
            if pivot.direction == "high"
        }
        atr = _atr_series(df)
        volume_available = volume_available or "volume" in df.columns

        for pivot in fractals:
            if pivot.direction != "high":
                continue
            later = df.iloc[
                pivot.bar_position + 1 : pivot.bar_position + 1 + reaction_bars
            ]
            drop = (
                max(0.0, pivot.price - float(later["low"].min()))
                if not later.empty
                else 0.0
            )
            pivot_atr = max(float(atr.iloc[pivot.bar_position]), 1e-9)
            evidence.append(
                PeakEvidence(
                    timeframe=timeframe,
                    timestamp=pivot.timestamp,
                    price=pivot.price,
                    age_bars=len(df) - 1 - pivot.bar_position,
                    zigzag_confirmed=pivot.timestamp in confirmed_zigzag_highs,
                    reaction_atr=drop / pivot_atr,
                    volume_ratio=_volume_ratio_at(df, pivot.bar_position),
                )
            )
    return evidence, volume_available


def _cluster_evidence(
    evidence: list[PeakEvidence],
    tolerance: float,
) -> list[list[PeakEvidence]]:
    clusters: list[list[PeakEvidence]] = []
    for item in sorted(evidence, key=lambda value: value.price):
        best_cluster = None
        best_distance = float("inf")
        for cluster in clusters:
            center = sum(point.price for point in cluster) / len(cluster)
            distance = abs(item.price - center)
            if distance <= tolerance and distance < best_distance:
                best_cluster = cluster
                best_distance = distance
        if best_cluster is None:
            clusters.append([item])
        else:
            best_cluster.append(item)
    return clusters


def _reliability(score: int) -> str:
    if score >= 12:
        return "RẤT CAO"
    if score >= 9:
        return "CAO"
    if score >= 6:
        return "TRUNG BÌNH"
    return "THẤP"


def _refresh_zone_position(
    zone: PeakZone,
    current_price: float,
    latest_closed_15m: float,
) -> None:
    if zone.lower <= current_price <= zone.upper:
        zone.status = "ĐANG TEST"
        zone.distance = 0.0
    elif zone.center > current_price:
        zone.status = "CẢN TRÊN"
        zone.distance = max(0.0, zone.lower - current_price)
    else:
        zone.status = "ĐỈNH ĐÃ VƯỢT"
        zone.distance = max(0.0, current_price - zone.upper)
    zone.support_confirmed = latest_closed_15m > zone.upper


def _zone_from_cluster(
    cluster: list[PeakEvidence],
    current_price: float,
    padding: float,
    latest_closed_15m: float,
    new_peak_bars: int,
) -> PeakZone:
    weights = [
        TIMEFRAME_WEIGHTS.get(point.timeframe, 1)
        * (1.25 if point.zigzag_confirmed else 1.0)
        for point in cluster
    ]
    center = sum(
        point.price * weight for point, weight in zip(cluster, weights)
    ) / sum(weights)
    lower = min(point.price for point in cluster) - padding
    upper = max(point.price for point in cluster) + padding
    timeframes = tuple(
        sorted(
            {point.timeframe for point in cluster},
            key=lambda timeframe: TIMEFRAME_WEIGHTS.get(timeframe, 0),
            reverse=True,
        )
    )
    newest = max(cluster, key=lambda point: point.timestamp)
    zigzag_count = sum(point.zigzag_confirmed for point in cluster)
    reaction_atr = max(point.reaction_atr for point in cluster)
    volume_spike = any(
        point.volume_ratio is not None and point.volume_ratio >= 1.5
        for point in cluster
    )

    score = max(TIMEFRAME_WEIGHTS.get(point.timeframe, 1) for point in cluster)
    score += min(3, len(cluster))
    score += min(3, max(0, len(timeframes) - 1) * 2)
    score += 2 if zigzag_count else 0
    score += 2 if reaction_atr >= 2 else 1 if reaction_atr >= 1 else 0
    score += 2 if volume_spike else 0

    if lower <= current_price <= upper:
        status = "ĐANG TEST"
        distance = 0.0
    elif center > current_price:
        status = "CẢN TRÊN"
        distance = max(0.0, lower - current_price)
    else:
        status = "ĐỈNH ĐÃ VƯỢT"
        distance = max(0.0, current_price - upper)

    return PeakZone(
        lower=lower,
        upper=upper,
        center=center,
        reliability=_reliability(score),
        score=score,
        age_label="MỚI" if newest.age_bars <= new_peak_bars else "CŨ",
        newest_at=newest.timestamp,
        timeframes=timeframes,
        evidence_count=len(cluster),
        zigzag_count=zigzag_count,
        reaction_atr=reaction_atr,
        volume_spike=volume_spike,
        status=status,
        distance=distance,
        support_confirmed=latest_closed_15m > upper,
    )


def build_peak_map(
    frames: dict[str, pd.DataFrame],
    current_price: float,
    settings: dict | None = None,
) -> PeakMap:
    settings = settings or {}
    evidence, volume_available = _build_peak_evidence(frames, settings)
    max_distance_pct = float(settings.get("max_distance_pct", 5.0)) / 100
    evidence = [
        point
        for point in evidence
        if abs(point.price - current_price) / max(current_price, 1e-9)
        <= max_distance_pct
    ]

    base = frames.get("15min")
    if base is None:
        base = next(iter(frames.values()))
    base_atr = float(_atr_series(base).iloc[-1])
    tolerance = max(
        current_price * float(settings.get("cluster_tolerance_pct", 0.04)) / 100,
        base_atr * 0.5,
    )
    padding = max(
        current_price * float(settings.get("zone_padding_pct", 0.02)) / 100,
        base_atr * 0.25,
    )
    latest_closed_15m = float(base["close"].iloc[-1])
    new_peak_bars = max(2, int(settings.get("new_peak_bars", 12)))
    zones = [
        _zone_from_cluster(
            cluster,
            current_price,
            padding,
            latest_closed_15m,
            new_peak_bars,
        )
        for cluster in _cluster_evidence(evidence, tolerance)
    ]
    minimum_zone_score = max(1, int(settings.get("minimum_zone_score", 6)))
    zones = [zone for zone in zones if zone.score >= minimum_zone_score]
    # Expanded ATR/percentage bands from adjacent clusters must not overlap.
    ordered_zones = sorted(zones, key=lambda zone: zone.center)
    for left, right in zip(ordered_zones, ordered_zones[1:]):
        if left.upper >= right.lower:
            boundary = (left.center + right.center) / 2
            left.upper = min(left.upper, boundary)
            right.lower = max(right.lower, boundary)
    for zone in ordered_zones:
        _refresh_zone_position(zone, current_price, latest_closed_15m)

    resistance = [
        zone
        for zone in zones
        if zone.center >= current_price or zone.status == "ĐANG TEST"
    ]
    converted_support = [
        zone
        for zone in zones
        if zone.upper < current_price and zone.support_confirmed
    ]
    resistance.sort(
        key=lambda zone: (
            0 if zone.status == "ĐANG TEST" else 1,
            zone.distance,
            -zone.score,
        )
    )
    converted_support.sort(key=lambda zone: (zone.distance, -zone.score))

    return PeakMap(
        current_price=current_price,
        resistance_zones=resistance,
        converted_support_zones=converted_support,
        volume_available=volume_available,
        scanned_peak_count=len(evidence),
    )


def assess_peak_liquidity(
    frame_1h: pd.DataFrame,
    current_price: float,
    bid_price: float | None,
    ask_price: float | None,
    analysis_now: datetime | None = None,
    settings: dict | None = None,
) -> PeakLiquidityAssessment:
    """Compare recent XAUUSDT volume with its weekday baseline and guard weekends."""
    settings = settings or {}
    analysis_now = analysis_now or datetime.now(timezone.utc)
    if analysis_now.tzinfo is None:
        analysis_now = analysis_now.replace(tzinfo=timezone.utc)
    is_weekend = is_weekend_entry_blackout(
        analysis_now,
        settings.get("weekend_cutoff_hour_vn", 5),
    )

    volume_column = (
        "quote_volume"
        if "quote_volume" in frame_1h.columns
        else "volume"
        if "volume" in frame_1h.columns
        else None
    )
    recent_volume = None
    baseline_volume = None
    volume_ratio = None
    if volume_column is not None:
        volumes = pd.to_numeric(frame_1h[volume_column], errors="coerce").dropna()
        recent_bars = max(3, int(settings.get("recent_volume_hours", 6)))
        recent = volumes.tail(recent_bars)
        prior = volumes.iloc[: -recent_bars].tail(
            max(24, int(settings.get("volume_baseline_hours", 168)))
        )
        if not prior.empty:
            weekday_prior = prior[prior.index.dayofweek < 5]
            recent_clock_hours = set(recent.index.hour)
            same_session_weekdays = weekday_prior[
                weekday_prior.index.hour.isin(recent_clock_hours)
            ]
            if len(same_session_weekdays) >= 12:
                baseline_source = same_session_weekdays
            elif len(weekday_prior) >= 12:
                baseline_source = weekday_prior
            else:
                baseline_source = prior
            recent_volume = float(recent.median()) if not recent.empty else None
            baseline_volume = float(baseline_source.median())
            if recent_volume is not None and baseline_volume > 0:
                volume_ratio = recent_volume / baseline_volume

    spread_bps = None
    if (
        bid_price is not None
        and ask_price is not None
        and ask_price >= bid_price
        and current_price > 0
    ):
        spread_bps = (ask_price - bid_price) / current_price * 10_000

    minimum_volume_ratio = max(
        0.0,
        float(settings.get("minimum_volume_ratio", 0.45)),
    )
    maximum_spread_bps = max(
        0.0,
        float(settings.get("maximum_spread_bps", 3.0)),
    )
    block_weekend = bool(settings.get("block_weekend_entries", True))

    if is_weekend and block_weekend:
        entries_allowed = False
        status = "THẤP / CUỐI TUẦN"
        ratio_text = (
            f"; volume 1H gần đây bằng {volume_ratio:.0%} median ngày thường"
            if volume_ratio is not None
            else ""
        )
        reason = (
            "Cuối tuần: thị trường tham chiếu truyền thống nghỉ, bot chỉ vẽ vùng "
            f"và không phát Entry{ratio_text}."
        )
    elif volume_ratio is not None and volume_ratio < minimum_volume_ratio:
        entries_allowed = False
        status = "THẤP"
        reason = (
            f"Thanh khoản 1H chỉ bằng {volume_ratio:.0%} median ngày thường, "
            "không đủ để xác nhận phá/retest."
        )
    elif spread_bps is not None and spread_bps > maximum_spread_bps:
        entries_allowed = False
        status = "SPREAD RỘNG"
        reason = (
            f"Spread {spread_bps:.2f} bps vượt ngưỡng {maximum_spread_bps:.2f} bps."
        )
    else:
        entries_allowed = True
        status = "TỐT" if volume_ratio is not None and volume_ratio >= 0.8 else "BÌNH THƯỜNG"
        ratio_text = (
            f"Volume 1H bằng {volume_ratio:.0%} median ngày thường."
            if volume_ratio is not None
            else "Chưa đủ lịch sử để so volume ngày thường."
        )
        reason = ratio_text

    return PeakLiquidityAssessment(
        is_weekend=is_weekend,
        status=status,
        recent_hourly_volume=recent_volume,
        weekday_baseline_volume=baseline_volume,
        volume_ratio=volume_ratio,
        spread_bps=spread_bps,
        entries_allowed=entries_allowed,
        reason=reason,
    )


def assess_peak_trade_gate(
    peak_map: PeakMap,
    frame_15m: pd.DataFrame,
    momentum_scores: dict[str, float],
    settings: dict | None = None,
    frame_1h: pd.DataFrame | None = None,
    liquidity: PeakLiquidityAssessment | None = None,
    daily_pattern: str | None = None,
) -> PeakTradeGate:
    """Allow a directional AI review only after closed-bar structural confirmation."""
    settings = settings or {}
    resistance = peak_map.resistance_zones[0] if peak_map.resistance_zones else None
    support = (
        peak_map.converted_support_zones[0]
        if peak_map.converted_support_zones
        else None
    )
    if resistance is None:
        return PeakTradeGate(
            "CHỜ",
            "Không có vùng cản trên đã xác nhận đủ gần.",
            None,
            support,
            False,
            False,
            False,
        )

    recent = frame_15m.tail(max(4, int(settings.get("ai_review_bars", 8))))
    atr = max(float(_atr_series(frame_15m).iloc[-1]), peak_map.current_price * 0.0001)
    tolerance = max(atr * 0.15, peak_map.current_price * 0.0001)

    breakout_at = None
    for position in range(1, len(recent)):
        if (
            float(recent["close"].iloc[position - 1]) <= resistance.upper
            and float(recent["close"].iloc[position]) > resistance.upper
        ):
            breakout_at = position
    long_retest = False
    if breakout_at is not None and breakout_at + 1 < len(recent):
        after_breakout = recent.iloc[breakout_at + 1 :]
        long_retest = bool(
            (
                (after_breakout["low"] <= resistance.upper + tolerance)
                & (after_breakout["low"] >= resistance.lower - tolerance)
                & (after_breakout["close"] > resistance.upper)
            ).any()
        )

    latest_two = recent.tail(2)
    candle_range = (latest_two["high"] - latest_two["low"]).clip(lower=1e-9)
    body = (latest_two["close"] - latest_two["open"]).abs()
    upper_wick = latest_two["high"] - latest_two[["open", "close"]].max(axis=1)
    touched = (latest_two["high"] >= resistance.lower) & (
        latest_two["low"] <= resistance.upper
    )
    short_rejection = bool(
        (
            touched
            & (latest_two["close"] < resistance.lower)
            & (latest_two["close"] < latest_two["open"])
            & ((upper_wick >= body) | (body / candle_range >= 0.55))
        ).any()
    )

    required = ("15min", "1h", "4h")
    minimum_score = float(settings.get("minimum_timeframe_score", 0.12))
    long_aligned = all(
        momentum_scores.get(timeframe, 0.0) >= minimum_score
        for timeframe in required
    )
    short_aligned = all(
        momentum_scores.get(timeframe, 0.0) <= -minimum_score
        for timeframe in required
    )
    current_near_resistance = (
        resistance.lower - atr * 0.30
        <= peak_map.current_price
        <= resistance.upper + atr * 0.30
    )

    require_hourly_close = bool(
        settings.get("require_1h_close_confirmation", True)
    )
    if frame_1h is not None and not frame_1h.empty:
        hourly_latest = frame_1h.iloc[-1]
        hourly_recent = frame_1h.tail(2)
        hourly_long_confirmed = float(hourly_latest["close"]) > resistance.upper
        hourly_short_confirmed = bool(
            float(hourly_latest["close"]) < resistance.lower
            and float(hourly_latest["close"]) < float(hourly_latest["open"])
            and (hourly_recent["high"] >= resistance.lower).any()
        )
    else:
        hourly_long_confirmed = not require_hourly_close
        hourly_short_confirmed = not require_hourly_close
    liquidity_confirmed = liquidity is None or liquidity.entries_allowed
    require_daily_context = bool(settings.get("require_daily_context", True))
    daily_opposition_limit = max(
        0.0,
        float(settings.get("daily_opposition_limit", 0.15)),
    )
    daily_score = momentum_scores.get("1day")
    daily_long_confirmed = (
        daily_score is not None and daily_score >= -daily_opposition_limit
        and daily_pattern != "LH/LL"
    ) or not require_daily_context
    daily_short_confirmed = (
        daily_score is not None and daily_score <= daily_opposition_limit
        and daily_pattern != "HH/HL"
    ) or not require_daily_context

    if (
        long_retest
        and long_aligned
        and hourly_long_confirmed
        and daily_long_confirmed
        and liquidity_confirmed
        and current_near_resistance
    ):
        return PeakTradeGate(
            "CANH LONG",
            "Ba khung cùng tăng; 15m phá/retest giữ cản và nến 1H đóng xác nhận.",
            resistance,
            support,
            True,
            short_rejection,
            True,
            True,
            True,
            True,
        )
    if (
        short_rejection
        and short_aligned
        and hourly_short_confirmed
        and daily_short_confirmed
        and liquidity_confirmed
        and current_near_resistance
    ):
        return PeakTradeGate(
            "CANH SHORT",
            "Ba khung cùng giảm; 15m từ chối cản và nến 1H đóng xác nhận.",
            resistance,
            support,
            long_retest,
            True,
            True,
            True,
            True,
            True,
        )

    if not liquidity_confirmed:
        reason = liquidity.reason
    elif not (long_aligned or short_aligned):
        reason = "Ba khung 15m/1H/4H chưa đồng thuận cùng hướng."
    elif long_aligned and not long_retest:
        reason = "Động lượng nghiêng tăng nhưng chưa có breakout và retest 15m hợp lệ."
    elif long_aligned and require_hourly_close and not hourly_long_confirmed:
        reason = f"15m đã nghiêng LONG nhưng nến 1H chưa đóng trên {resistance.upper:.2f}."
    elif long_aligned and not daily_long_confirmed:
        reason = (
            f"D1 {daily_pattern or '?'} đang nghiêng giảm {daily_score * 100:+.0f}%, "
            "không mở LONG ngược xu hướng lớn."
        )
    elif short_aligned and not short_rejection:
        reason = "Động lượng nghiêng giảm nhưng chưa có nến 15m từ chối cản hợp lệ."
    elif short_aligned and require_hourly_close and not hourly_short_confirmed:
        reason = f"15m đã nghiêng SHORT nhưng nến 1H chưa đóng từ chối dưới {resistance.lower:.2f}."
    elif short_aligned and not daily_short_confirmed:
        reason = (
            f"D1 {daily_pattern or '?'} đang nghiêng tăng {daily_score * 100:+.0f}%, "
            "không mở SHORT ngược xu hướng lớn."
        )
    else:
        reason = "Xác nhận có nhưng giá hiện đã rời vùng, không đuổi giá."
    return PeakTradeGate(
        "CHỜ",
        reason,
        resistance,
        support,
        long_retest,
        short_rejection,
        long_aligned or short_aligned,
        (
            hourly_long_confirmed
            if long_aligned
            else hourly_short_confirmed
            if short_aligned
            else False
        ),
        liquidity_confirmed,
        (
            daily_long_confirmed
            if long_aligned
            else daily_short_confirmed
            if short_aligned
            else False
        ),
    )


def build_peak_execution_plan(
    peak_map: PeakMap,
    gate: PeakTradeGate,
    frame_15m: pd.DataFrame,
    settings: dict | None = None,
) -> tuple[PeakExecutionPlan | None, str]:
    """Build levels only for a code-confirmed setup with a viable structural target."""
    settings = settings or {}
    if gate.allowed_decision not in ("CANH LONG", "CANH SHORT"):
        return None, gate.reason
    if gate.resistance is None:
        return None, "Không có vùng cản gốc để tính điểm vô hiệu."

    tick = max(float(settings.get("price_tick", 0.01)), 1e-9)
    atr = max(float(_atr_series(frame_15m).iloc[-1]), peak_map.current_price * 0.0001)
    entry_padding = max(
        atr * float(settings.get("entry_buffer_atr", 0.15)),
        tick,
    )
    stop_padding = max(
        atr * float(settings.get("stop_buffer_atr", 0.20)),
        tick * 2,
    )
    minimum_structural_rr = max(
        1.0,
        float(settings.get("minimum_structural_rr", 1.5)),
    )
    tp1_r = max(0.5, float(settings.get("take_profit_1_r", 1.0)))
    resistance = gate.resistance

    def rounded(value: float) -> float:
        return round(value / tick) * tick

    if gate.allowed_decision == "CANH LONG":
        side = "LONG"
        entry_lower = rounded(resistance.upper)
        entry_upper = rounded(resistance.upper + entry_padding)
        stop_loss = rounded(resistance.lower - stop_padding)
        higher_resistance = sorted(
            (
                zone
                for zone in peak_map.resistance_zones
                if zone.lower > entry_upper + tick
            ),
            key=lambda zone: zone.lower,
        )
        if not higher_resistance:
            return None, "Không có cản trên kế tiếp để đặt TP cấu trúc an toàn."
        target_zone = higher_resistance[0]
        take_profit_2 = rounded(target_zone.lower)
        structural_target = (
            f"cản kế tiếp {target_zone.lower:.2f}–{target_zone.upper:.2f}"
        )
        direction = 1
    else:
        side = "SHORT"
        entry_lower = rounded(resistance.lower - entry_padding)
        entry_upper = rounded(resistance.lower)
        stop_loss = rounded(resistance.upper + stop_padding)
        lower_support = sorted(
            (
                zone
                for zone in peak_map.converted_support_zones
                if zone.upper < entry_lower - tick
            ),
            key=lambda zone: zone.upper,
            reverse=True,
        )
        if not lower_support:
            return None, "Không có hỗ trợ phía dưới để đặt TP cấu trúc an toàn."
        target_zone = lower_support[0]
        take_profit_2 = rounded(target_zone.upper)
        structural_target = (
            f"hỗ trợ kế tiếp {target_zone.lower:.2f}–{target_zone.upper:.2f}"
        )
        direction = -1

    entry_reference = rounded((entry_lower + entry_upper) / 2)
    risk_distance = direction * (entry_reference - stop_loss)
    reward_2 = direction * (take_profit_2 - entry_reference)
    if risk_distance <= 0 or reward_2 <= 0:
        return None, "Vùng Entry/SL/TP không đúng thứ tự giá; bỏ thiết lập này."

    reward_risk_2 = reward_2 / risk_distance
    if reward_risk_2 < minimum_structural_rr:
        return (
            None,
            f"Bỏ lệnh: mục tiêu cấu trúc chỉ đạt R:R {reward_risk_2:.2f}, "
            f"thấp hơn {minimum_structural_rr:.2f}.",
        )

    take_profit_1 = rounded(
        entry_reference + direction * risk_distance * tp1_r
    )
    reward_risk_1 = (
        direction * (take_profit_1 - entry_reference) / risk_distance
    )
    if direction * (take_profit_2 - take_profit_1) <= 0:
        return None, "TP cấu trúc nằm trước TP1; tỷ lệ lời/lỗ không đủ để vào."

    return (
        PeakExecutionPlan(
            side=side,
            entry_lower=min(entry_lower, entry_upper),
            entry_upper=max(entry_lower, entry_upper),
            entry_reference=entry_reference,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            reward_risk_1=reward_risk_1,
            reward_risk_2=reward_risk_2,
            structural_target=structural_target,
        ),
        "Thiết lập đã qua xác nhận nến đóng và bộ lọc R:R cấu trúc.",
    )


def render_peak_hourly_chart(
    frame_1h: pd.DataFrame,
    peak_map: PeakMap,
    liquidity: PeakLiquidityAssessment | None = None,
) -> BytesIO:
    """Render a dedicated closed-candle 1H chart with peak zones and real volume."""
    data = frame_1h.tail(72).copy()
    if len(data) < 5:
        raise ValueError("Need at least 5 closed 1H candles to render peak chart")

    fig, (price_ax, volume_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )
    fig.patch.set_facecolor("#10151d")
    price_ax.set_facecolor("#10151d")
    volume_ax.set_facecolor("#10151d")
    width = 0.62
    candle_colors = []
    for x, (_, row) in enumerate(data.iterrows()):
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        candle_colors.append(color)
        price_ax.vlines(x, row["low"], row["high"], color=color, linewidth=0.8)
        bottom = min(row["open"], row["close"])
        height = max(abs(row["close"] - row["open"]), peak_map.current_price * 0.000005)
        price_ax.add_patch(
            plt.Rectangle(
                (x - width / 2, bottom),
                width,
                height,
                color=color,
                alpha=0.92,
            )
        )

    ema9 = EMAIndicator(data["close"], window=9).ema_indicator()
    ema21 = EMAIndicator(data["close"], window=21).ema_indicator()
    price_ax.plot(range(len(data)), ema9, color="#ffd166", linewidth=1, label="EMA9")
    price_ax.plot(range(len(data)), ema21, color="#4cc9f0", linewidth=1, label="EMA21")
    visible_low = min(float(data["low"].min()), peak_map.current_price)
    visible_high = max(float(data["high"].max()), peak_map.current_price)
    visible_padding = max((visible_high - visible_low) * 0.20, peak_map.current_price * 0.001)
    visible_resistance = [
        zone
        for zone in peak_map.resistance_zones
        if zone.lower <= visible_high + visible_padding
        and zone.upper >= visible_low - visible_padding
    ][:3]
    visible_support = [
        zone
        for zone in peak_map.converted_support_zones
        if zone.lower <= visible_high + visible_padding
        and zone.upper >= visible_low - visible_padding
    ][:2]
    for index, zone in enumerate(visible_resistance):
        price_ax.axhspan(
            zone.lower,
            zone.upper,
            color="#ff8c42",
            alpha=0.14,
            label="Kháng cự" if index == 0 else None,
        )
    for index, zone in enumerate(visible_support):
        price_ax.axhspan(
            zone.lower,
            zone.upper,
            color="#43aa8b",
            alpha=0.14,
            label="Hỗ trợ retest" if index == 0 else None,
        )
    price_ax.axhline(
        peak_map.current_price,
        color="#f8f9fa",
        linestyle="-.",
        linewidth=0.9,
        label=f"Live {peak_map.current_price:.2f}",
    )

    volume_column = "quote_volume" if "quote_volume" in data.columns else "volume"
    volume_values = pd.to_numeric(data[volume_column], errors="coerce").fillna(0)
    volume_ax.bar(range(len(data)), volume_values, color=candle_colors, alpha=0.7, width=0.7)
    baseline = volume_values.rolling(20, min_periods=5).median()
    volume_ax.plot(range(len(data)), baseline, color="#ffd166", linewidth=0.8, label="Median Vol20")
    volume_ax.set_ylabel("Volume", color="#aab2bf", fontsize=8)

    tick_count = min(7, len(data))
    ticks = [
        round(i * (len(data) - 1) / max(1, tick_count - 1))
        for i in range(tick_count)
    ]
    volume_ax.set_xticks(ticks)
    volume_ax.set_xticklabels(
        [
            data.index[i].tz_convert("Asia/Bangkok").strftime("%d/%m %H:%M")
            for i in ticks
        ],
        color="#aab2bf",
        fontsize=8,
    )
    for ax in (price_ax, volume_ax):
        ax.tick_params(axis="y", colors="#aab2bf", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(alpha=0.10)
    price_ax.legend(
        loc="upper left",
        ncol=5,
        fontsize=7,
        facecolor="#18212f",
        labelcolor="white",
    )
    volume_ax.legend(
        loc="upper left",
        fontsize=7,
        facecolor="#18212f",
        labelcolor="white",
    )
    liquidity_text = f" · thanh khoản {liquidity.status}" if liquidity is not None else ""
    price_ax.set_title(
        f"Binance XAUUSDT · 1H nến đã đóng{liquidity_text}",
        color="white",
        fontsize=12,
        loc="left",
    )
    fig.tight_layout()
    output = BytesIO()
    output.name = "xauusdt_peak_1h.png"
    fig.savefig(output, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    output.seek(0)
    return output


def render_peak_confirmation_chart(
    frames: dict[str, pd.DataFrame],
    peak_map: PeakMap,
    liquidity: PeakLiquidityAssessment | None = None,
) -> BytesIO:
    """Render the exact 15m Entry, 1H confirmation, and D1 context charts."""
    chart_specs = [
        ("15min", "15m · tìm Entry/retest", 64),
        ("1h", "1H · nến đóng xác nhận", 72),
        ("1day", "D1 · xu hướng lớn", 90),
    ]
    available = [spec for spec in chart_specs if spec[0] in frames]
    if len(available) != len(chart_specs):
        raise ValueError("Peak confirmation chart requires 15min, 1h, and 1day frames")

    fig, axes = plt.subplots(len(available), 1, figsize=(10, 11))
    fig.patch.set_facecolor("#10151d")
    for ax, (timeframe, label, lookback) in zip(axes, available):
        data = frames[timeframe].tail(lookback).copy()
        ax.set_facecolor("#10151d")
        width = 0.62
        candle_colors = []
        for x, (_, row) in enumerate(data.iterrows()):
            color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
            candle_colors.append(color)
            ax.vlines(x, row["low"], row["high"], color=color, linewidth=0.75)
            bottom = min(row["open"], row["close"])
            height = max(
                abs(row["close"] - row["open"]),
                peak_map.current_price * 0.000005,
            )
            ax.add_patch(
                plt.Rectangle(
                    (x - width / 2, bottom),
                    width,
                    height,
                    color=color,
                    alpha=0.92,
                )
            )

        ema9 = EMAIndicator(data["close"], window=9).ema_indicator()
        ema21 = EMAIndicator(data["close"], window=21).ema_indicator()
        ax.plot(range(len(data)), ema9, color="#ffd166", linewidth=0.9, label="EMA9")
        ax.plot(range(len(data)), ema21, color="#4cc9f0", linewidth=0.9, label="EMA21")

        visible_low = min(float(data["low"].min()), peak_map.current_price)
        visible_high = max(float(data["high"].max()), peak_map.current_price)
        visible_padding = max(
            (visible_high - visible_low) * 0.20,
            peak_map.current_price * 0.001,
        )
        resistance = [
            zone
            for zone in peak_map.resistance_zones
            if zone.lower <= visible_high + visible_padding
            and zone.upper >= visible_low - visible_padding
        ][:3]
        support = [
            zone
            for zone in peak_map.converted_support_zones
            if zone.lower <= visible_high + visible_padding
            and zone.upper >= visible_low - visible_padding
        ][:2]
        for index, zone in enumerate(resistance):
            ax.axhspan(
                zone.lower,
                zone.upper,
                color="#ff8c42",
                alpha=0.13,
                label="Kháng cự" if index == 0 else None,
            )
        for index, zone in enumerate(support):
            ax.axhspan(
                zone.lower,
                zone.upper,
                color="#43aa8b",
                alpha=0.13,
                label="Hỗ trợ" if index == 0 else None,
            )
        ax.axhline(
            peak_map.current_price,
            color="#f8f9fa",
            linestyle="-.",
            linewidth=0.8,
            label=f"Live {peak_map.current_price:.2f}",
        )

        volume_column = "quote_volume" if "quote_volume" in data.columns else "volume"
        volume_values = pd.to_numeric(data[volume_column], errors="coerce").fillna(0)
        volume_ax = ax.twinx()
        volume_ax.bar(
            range(len(data)),
            volume_values,
            color=candle_colors,
            alpha=0.10,
            width=0.7,
        )
        max_volume = float(volume_values.max()) if not volume_values.empty else 0.0
        if max_volume > 0:
            volume_ax.set_ylim(0, max_volume * 4)
        volume_ax.set_yticks([])
        for spine in volume_ax.spines.values():
            spine.set_visible(False)

        tick_count = min(6, len(data))
        ticks = [
            round(i * (len(data) - 1) / max(1, tick_count - 1))
            for i in range(tick_count)
        ]
        time_format = "%d/%m" if timeframe == "1day" else "%d/%m %H:%M"
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [
                data.index[i].tz_convert("Asia/Bangkok").strftime(time_format)
                for i in ticks
            ],
            color="#aab2bf",
            fontsize=7,
        )
        ax.tick_params(axis="y", colors="#aab2bf", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(alpha=0.10)
        liquidity_text = (
            f" · thanh khoản {liquidity.status}"
            if timeframe == "1h" and liquidity is not None
            else ""
        )
        ax.set_title(
            label + liquidity_text,
            color="white",
            fontsize=10,
            loc="left",
        )
        ax.legend(
            loc="upper left",
            ncol=5,
            fontsize=6.5,
            facecolor="#18212f",
            labelcolor="white",
        )

    fig.suptitle(
        "Binance XAUUSDT · 15m Entry / 1H xác nhận / D1 xu hướng",
        color="white",
        fontsize=13,
    )
    fig.tight_layout()
    output = BytesIO()
    output.name = "xauusdt_confirmation_15m_1h_d1.png"
    fig.savefig(output, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    output.seek(0)
    return output


def format_peak_map(
    peak_map: PeakMap,
    settings: dict | None = None,
) -> str:
    settings = settings or {}
    max_resistance = max(1, int(settings.get("max_resistance_zones", 3)))
    max_support = max(1, int(settings.get("max_support_zones", 2)))

    def zone_lines(zone: PeakZone, index: int) -> list[str]:
        timeframe_text = "+".join(
            TIMEFRAME_LABELS.get(timeframe, timeframe)
            for timeframe in zone.timeframes
        )
        method = "Fractal+ZZ" if zone.zigzag_count else "Fractal"
        volume_mark = " · Vol≥1.5x" if zone.volume_spike else ""
        local_time = zone.newest_at.tz_convert("Asia/Bangkok").strftime("%d/%m %H:%M")
        return [
            f"{index}. *{zone.lower:.2f}–{zone.upper:.2f}* · {zone.reliability} · {zone.status}",
            f"   {zone.age_label} {timeframe_text} {local_time} · {method} · "
            f"{zone.evidence_count} đỉnh · phản ứng {zone.reaction_atr:.1f} ATR"
            f"{volume_mark}",
        ]

    def select_resistance(zones: list[PeakZone]) -> list[PeakZone]:
        if not zones:
            return []
        selected = [zones[0]]
        large_timeframe = sorted(
            (
                zone
                for zone in zones[1:]
                if any(
                    TIMEFRAME_WEIGHTS.get(timeframe, 0)
                    >= TIMEFRAME_WEIGHTS["1h"]
                    for timeframe in zone.timeframes
                )
            ),
            key=lambda zone: (-zone.score, zone.distance),
        )
        if large_timeframe and len(selected) < max_resistance:
            selected.append(large_timeframe[0])
        for zone in zones[1:]:
            if len(selected) >= max_resistance:
                break
            if zone not in selected:
                selected.append(zone)
        return sorted(
            selected,
            key=lambda zone: (
                0 if zone.status == "ĐANG TEST" else 1,
                zone.center,
            ),
        )

    lines = [
        f"⛰ *BẢN ĐỒ ĐỈNH BINANCE XAUUSDT {peak_map.current_price:.2f}*",
        "",
        "*CẢN TRÊN / ĐỈNH ĐANG TEST*",
    ]
    resistance = select_resistance(peak_map.resistance_zones)
    if resistance:
        for index, zone in enumerate(resistance, 1):
            lines.extend(zone_lines(zone, index))
    else:
        lines.append("Chưa có đỉnh xác nhận đủ gần giá hiện tại.")

    lines += ["", "*ĐỈNH CŨ ĐÃ VƯỢT → HỖ TRỢ RETEST*"]
    supports = peak_map.converted_support_zones[:max_support]
    if supports:
        for index, zone in enumerate(supports, 1):
            lines.extend(zone_lines(zone, index))
    else:
        lines.append("Chưa có đỉnh cũ phía dưới được nến 15m đóng vượt.")

    nearest = resistance[0] if resistance else None
    lines += ["", "*CÁCH DÙNG*"]
    if nearest is not None:
        lines.append(
            f"• SHORT: chỉ xét khi giá vào {nearest.lower:.2f}–{nearest.upper:.2f} "
            "và 15m/1H đóng từ chối dưới vùng, D1 không tăng mạnh."
        )
        lines.append(
            f"• LONG: cần 15m phá/retest, 1H đóng trên {nearest.upper:.2f} và D1 không giảm mạnh."
        )
    lines.append("• Râu nến xuyên vùng không tính là phá; không vào giữa hai vùng.")
    lines.append(
        "• Volume: có dữ liệu thật và được cộng điểm khi ≥1.5x trung bình."
        if peak_map.volume_available
        else "• Volume: nguồn hiện tại không có volume hợp lệ nên không cộng điểm."
    )
    lines.append(
        "_Uy tín là điểm hội tụ cấu trúc, không phải xác suất thắng; "
        "Fractal/ZigZag có độ trễ và vùng đỉnh không bảo đảm đảo chiều._"
    )
    return "\n".join(lines)
