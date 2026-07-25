from dataclasses import dataclass

import pandas as pd
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
        f"⛰ *BẢN ĐỒ ĐỈNH XAU/USD {peak_map.current_price:.2f}*",
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
            "và nến 15m đóng từ chối dưới vùng."
        )
        lines.append(
            f"• LONG: cần nến 15m đóng trên {nearest.upper:.2f}, rồi retest giữ được vùng."
        )
    lines.append("• Râu nến xuyên vùng không tính là phá; không vào giữa hai vùng.")
    lines.append(
        "• Volume: có dữ liệu thật và được cộng điểm khi ≥1.5x trung bình."
        if peak_map.volume_available
        else "• Volume: XAU/USD nguồn hiện tại không có volume đáng tin nên không cộng điểm."
    )
    lines.append(
        "_Uy tín là điểm hội tụ cấu trúc, không phải xác suất thắng; "
        "Fractal/ZigZag có độ trễ và vùng đỉnh không bảo đảm đảo chiều._"
    )
    return "\n".join(lines)
