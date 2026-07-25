import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from xml.etree import ElementTree

import matplotlib
import pandas as pd
import requests

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

HTTP_HEADERS = {"User-Agent": "gold-signal-tool/1.0 (personal market research)"}
logger = logging.getLogger("gold-query-bot.market-context")


@dataclass
class PricePressure:
    score: float
    label: str
    up_bars: int
    down_bars: int
    window_minutes: int


@dataclass
class OrderBookPressure:
    score: float
    label: str
    bid_notional: float
    ask_notional: float
    symbol: str


@dataclass
class TimeframeConsensus:
    score: float
    label: str
    aligned: bool
    actionable: bool
    scores: dict[str, float]


@dataclass
class NewsItem:
    source: str
    title: str
    published_at: datetime | None
    link: str


@dataclass
class NewsFetchResult:
    items: list[NewsItem]
    fetched_at: datetime
    source_errors: list[str]


@dataclass
class SwingPoint:
    timestamp: pd.Timestamp
    price: float


@dataclass
class ChartStructure:
    trend: str
    pattern: str
    support: float
    resistance: float
    swing_highs: list[SwingPoint]
    swing_lows: list[SwingPoint]


@dataclass
class ScenarioLevels:
    break_up: float
    upside_target_1: float
    upside_target_2: float
    break_down: float
    downside_target_1: float
    downside_target_2: float


@dataclass
class ResistanceZoneAnalysis:
    lower: float
    upper: float
    state: str
    distance: float
    touched_recently: bool
    candle_pattern: str
    rsi14: float
    ema21: float
    ema50: float
    ma_confluence: bool
    volume_ratio: float | None
    rejection_score: int
    verdict: str
    rejection_low: float


@dataclass
class PriceZone:
    lower: float
    upper: float


@dataclass
class RetestAssessment:
    support: PriceZone
    resistance: PriceZone
    long_phase: str
    short_phase: str
    long_retest_confirmed: bool
    short_retest_confirmed: bool
    actionable_side: str | None
    decision_reason: str
    entry_lower: float | None
    entry_upper: float | None
    invalidation: float | None


def interval_duration(interval: str) -> timedelta:
    if interval.endswith("min"):
        return timedelta(minutes=float(interval[:-3]))
    if interval.endswith("h"):
        return timedelta(hours=float(interval[:-1]))
    if interval.endswith("day"):
        return timedelta(days=float(interval[:-3]))
    raise ValueError(f"Unsupported candle interval: {interval}")


def select_closed_candles(
    df: pd.DataFrame,
    interval: str,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Return only completed candles; Twelve Data intraday timestamps mark bar starts."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    index = df.index
    if index.tz is None:
        index = index.tz_localize(timezone.utc)
    else:
        index = index.tz_convert(timezone.utc)
    closed = df.copy()
    closed.index = index
    closed = closed.loc[closed.index + interval_duration(interval) <= now]
    if closed.empty:
        raise ValueError(f"No completed {interval} candles available")
    return closed


def analyze_price_pressure(df: pd.DataFrame) -> PricePressure:
    """Price-action proxy only; XAU/USD spot data has no centralized order book/volume."""
    closes = df["close"].astype(float)
    changes = closes.diff().dropna()
    if changes.empty:
        return PricePressure(0.0, "khong du du lieu", 0, 0, len(df))

    gross_move = float(changes.abs().sum()) or 1e-9
    directional_efficiency = float(changes.sum()) / gross_move
    up_bars = int((changes > 0).sum())
    down_bars = int((changes < 0).sum())
    active_bars = max(1, up_bars + down_bars)
    bar_imbalance = (up_bars - down_bars) / active_bars
    score = max(-1.0, min(1.0, 0.65 * directional_efficiency + 0.35 * bar_imbalance))

    if score >= 0.35:
        label = "ap luc gia mua chiem uu the"
    elif score >= 0.1:
        label = "ap luc gia mua nhe"
    elif score <= -0.35:
        label = "ap luc gia ban chiem uu the"
    elif score <= -0.1:
        label = "ap luc gia ban nhe"
    else:
        label = "mua/ban can bang"
    return PricePressure(score, label, up_bars, down_bars, len(df))


def fetch_order_book_pressure(endpoint: str, symbol: str, limit: int = 100) -> OrderBookPressure | None:
    """Use a tokenized-gold order book as a proxy, never as XAU/USD's own order book."""
    try:
        response = requests.get(
            endpoint,
            params={"symbol": symbol, "limit": limit},
            headers=HTTP_HEADERS,
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        bids = payload.get("bids", [])
        asks = payload.get("asks", [])
        bid_notional = sum(float(price) * float(quantity) for price, quantity in bids)
        ask_notional = sum(float(price) * float(quantity) for price, quantity in asks)
        total = bid_notional + ask_notional
        if total <= 0:
            return None
        score = max(-1.0, min(1.0, (bid_notional - ask_notional) / total))
        if score >= 0.2:
            label = "bid day hon ask"
        elif score <= -0.2:
            label = "ask day hon bid"
        else:
            label = "order book can bang"
        return OrderBookPressure(score, label, bid_notional, ask_notional, symbol)
    except Exception:
        return None


def build_consensus(
    scores: dict[str, float],
    pressure_score: float,
    settings: dict | None = None,
) -> TimeframeConsensus:
    settings = settings or {}
    weights = {"15min": 0.20, "1h": 0.30, "4h": 0.40}
    total = sum(scores.get(tf, 0.0) * weight for tf, weight in weights.items())
    total += pressure_score * 0.10
    total = max(-1.0, min(1.0, total))

    minimum_timeframe_score = float(settings.get("minimum_timeframe_score", 0.12))
    signs = [
        1
        if scores.get(tf, 0) >= minimum_timeframe_score
        else -1
        if scores.get(tf, 0) <= -minimum_timeframe_score
        else 0
        for tf in weights
    ]
    non_neutral = [sign for sign in signs if sign]
    required_timeframes = (
        len(weights) if settings.get("require_all_timeframes", True) else 2
    )
    aligned = (
        len(non_neutral) >= required_timeframes
        and len(set(non_neutral)) == 1
    )
    actionable = aligned and abs(total) >= float(
        settings.get("actionable_threshold", 0.40)
    )

    if not actionable:
        label = "KHONG VAO LENH - cac khung chua dong thuan"
    elif total > 0:
        label = "uu tien LONG da khung"
    else:
        label = "uu tien SHORT da khung"
    return TimeframeConsensus(total, label, aligned, actionable, scores)


def fetch_market_news(
    feed_urls: list[dict],
    limit: int = 4,
) -> NewsFetchResult:
    items: list[NewsItem] = []
    errors: list[str] = []
    for feed in feed_urls:
        try:
            response = requests.get(feed["url"], headers=HTTP_HEADERS, timeout=10)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            for node in root.findall(".//item")[:limit]:
                title = (node.findtext("title") or "").strip()
                if feed.get("language") == "vi" and not any(
                    character in title
                    for character in "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
                ):
                    continue
                published = None
                raw_date = node.findtext("pubDate")
                if raw_date:
                    try:
                        published = parsedate_to_datetime(raw_date)
                    except (TypeError, ValueError):
                        pass
                items.append(
                    NewsItem(
                        source=feed["name"],
                        title=title,
                        published_at=published,
                        link=(node.findtext("link") or "").strip(),
                    )
                )
        except Exception as exc:
            message = f"{feed.get('name', 'unknown')}: {type(exc).__name__}: {exc}"
            errors.append(message)
            logger.warning("News source failed: %s", message)

    def sort_key(item: NewsItem) -> datetime:
        value = item.published_at or datetime.min.replace(tzinfo=timezone.utc)
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    sorted_items = sorted(items, key=sort_key, reverse=True)
    unique_items: list[NewsItem] = []
    seen_titles: set[str] = set()
    for item in sorted_items:
        normalized_title = " ".join(item.title.lower().split())
        if normalized_title in seen_titles:
            continue
        seen_titles.add(normalized_title)
        unique_items.append(item)
    items = unique_items[:limit]
    now = datetime.now(timezone.utc)
    return NewsFetchResult(items, now, errors)


def news_risk_label(items: list[NewsItem]) -> str:
    keywords = (
        "federal reserve",
        "fed ",
        "fomc",
        "rate",
        "inflation",
        "cpi",
        "payroll",
        "jobs",
        "war",
        "conflict",
        "sanction",
        "tariff",
    )
    recent_titles = " ".join(item.title.lower() for item in items)
    return "CAO - tranh vao ngay luc tin ra" if any(word in recent_titles for word in keywords) else "binh thuong"


def find_chart_structure(df: pd.DataFrame, window: int = 3, lookback: int = 100) -> ChartStructure:
    data = df.tail(lookback)
    highs: list[SwingPoint] = []
    lows: list[SwingPoint] = []
    for index in range(window, len(data) - window):
        row = data.iloc[index]
        timestamp = data.index[index]
        area = data.iloc[index - window : index + window + 1]
        if row["high"] >= area["high"].max():
            highs.append(SwingPoint(timestamp, float(row["high"])))
        if row["low"] <= area["low"].min():
            lows.append(SwingPoint(timestamp, float(row["low"])))

    recent_highs = highs[-3:]
    recent_lows = lows[-3:]
    resistance = recent_highs[-1].price if recent_highs else float(data["high"].tail(20).max())
    support = recent_lows[-1].price if recent_lows else float(data["low"].tail(20).min())

    high_pattern = "?"
    low_pattern = "?"
    if len(recent_highs) >= 2:
        high_pattern = "HH" if recent_highs[-1].price > recent_highs[-2].price else "LH"
    if len(recent_lows) >= 2:
        low_pattern = "HL" if recent_lows[-1].price > recent_lows[-2].price else "LL"

    pattern = f"{high_pattern}/{low_pattern}"
    if high_pattern == "HH" and low_pattern == "HL":
        trend = "tang (dinh cao hon, day cao hon)"
    elif high_pattern == "LH" and low_pattern == "LL":
        trend = "giam (dinh thap hon, day thap hon)"
    else:
        trend = "hon hop/di ngang"
    return ChartStructure(trend, pattern, support, resistance, recent_highs, recent_lows)


def build_scenario_levels(
    current_price: float,
    atr: float,
    structures: dict[str, ChartStructure],
) -> ScenarioLevels:
    """Build monotonic scenario levels from detected pivots, with ATR fallbacks."""
    atr = max(float(atr), current_price * 0.0005)
    minimum_gap = atr * 0.35
    candidates: list[float] = []
    for structure in structures.values():
        candidates.extend([structure.support, structure.resistance])
        candidates.extend(point.price for point in structure.swing_highs)
        candidates.extend(point.price for point in structure.swing_lows)

    def spaced(values: list[float], reverse: bool = False) -> list[float]:
        ordered = sorted(set(round(value, 5) for value in values), reverse=reverse)
        result: list[float] = []
        for value in ordered:
            if not result or abs(value - result[-1]) >= minimum_gap:
                result.append(value)
        return result

    above = spaced([value for value in candidates if value > current_price + minimum_gap])
    below = spaced(
        [value for value in candidates if value < current_price - minimum_gap],
        reverse=True,
    )

    break_up = (
        above[0]
        if above and above[0] <= current_price + 2 * atr
        else current_price + atr * 0.75
    )
    higher = [
        value
        for value in above
        if break_up + minimum_gap <= value <= break_up + 2 * atr
    ]
    upside_target_1 = higher[0] if higher else break_up + atr
    higher_2 = [
        value
        for value in above
        if upside_target_1 + minimum_gap <= value <= upside_target_1 + 2 * atr
    ]
    upside_target_2 = higher_2[0] if higher_2 else upside_target_1 + atr

    break_down = (
        below[0]
        if below and below[0] >= current_price - 2 * atr
        else current_price - atr * 0.75
    )
    lower = [
        value
        for value in below
        if break_down - 2 * atr <= value <= break_down - minimum_gap
    ]
    downside_target_1 = lower[0] if lower else break_down - atr
    lower_2 = [
        value
        for value in below
        if downside_target_1 - 2 * atr <= value <= downside_target_1 - minimum_gap
    ]
    downside_target_2 = lower_2[0] if lower_2 else downside_target_1 - atr

    return ScenarioLevels(
        break_up=break_up,
        upside_target_1=upside_target_1,
        upside_target_2=upside_target_2,
        break_down=break_down,
        downside_target_1=downside_target_1,
        downside_target_2=downside_target_2,
    )


def analyze_resistance_zone(
    df: pd.DataFrame,
    current_price: float,
    atr: float,
    structures: dict[str, ChartStructure],
    settings: dict | None = None,
) -> ResistanceZoneAnalysis:
    """Evaluate the nearest resistance as an ATR-sized zone, not a single exact price."""
    settings = settings or {}
    data = df.tail(100).copy()
    atr = max(float(atr), current_price * 0.0005)
    cluster_tolerance = atr * float(settings.get("cluster_atr", 0.35))
    padding = max(
        atr * float(settings.get("padding_atr", 0.12)),
        current_price * float(settings.get("minimum_padding_pct", 0.00015)),
    )

    levels: list[float] = []
    for structure in structures.values():
        levels.append(float(structure.resistance))
        levels.extend(float(point.price) for point in structure.swing_highs)
    levels = [
        value
        for value in levels
        if pd.notna(value)
        and current_price - atr * 0.6 <= value <= current_price + atr * 4
    ]
    if not levels:
        levels = [float(data["high"].tail(20).max())]

    seed = min(levels, key=lambda value: abs(value - current_price))
    cluster = [
        value for value in levels if abs(value - seed) <= cluster_tolerance
    ] or [seed]
    lower = min(cluster) - padding
    upper = max(cluster) + padding

    if current_price < lower:
        state = "ĐANG TIẾN GẦN"
        distance = lower - current_price
    elif current_price <= upper:
        state = "ĐANG TEST VÙNG"
        distance = 0.0
    else:
        state = "ĐANG Ở TRÊN VÙNG - CHỜ NẾN ĐÓNG XÁC NHẬN"
        distance = current_price - upper

    recent = data.tail(3)
    touched_recently = bool(
        ((recent["high"] >= lower) & (recent["low"] <= upper)).any()
    )
    latest = data.iloc[-1]
    previous = data.iloc[-2]
    candle_range = max(float(latest["high"] - latest["low"]), 1e-9)
    body = abs(float(latest["close"] - latest["open"]))
    upper_wick = float(latest["high"] - max(latest["open"], latest["close"]))
    lower_wick = float(min(latest["open"], latest["close"]) - latest["low"])
    body_ratio = body / candle_range
    upper_wick_ratio = upper_wick / candle_range
    latest_touched = bool(
        float(latest["high"]) >= lower and float(latest["low"]) <= upper
    )
    bearish_engulfing = bool(
        latest_touched
        and previous["close"] > previous["open"]
        and latest["close"] < latest["open"]
        and latest["open"] >= previous["close"]
        and latest["close"] <= previous["open"]
    )
    shooting_star = bool(
        latest_touched
        and upper_wick >= max(body * 2, candle_range * 0.42)
        and lower_wick <= candle_range * 0.2
        and body_ratio <= 0.4
    )
    doji = bool(latest_touched and body_ratio <= 0.2)
    upper_wick_rejection = bool(
        latest_touched
        and upper_wick_ratio >= 0.4
        and latest["close"] <= latest["low"] + candle_range * 0.58
    )
    long_bearish = bool(
        latest_touched
        and latest["close"] < latest["open"]
        and body_ratio >= 0.6
        and latest["close"] < lower
    )

    if bearish_engulfing:
        candle_pattern = "Bearish Engulfing"
        pattern_score = 2
    elif shooting_star:
        candle_pattern = "Shooting Star"
        pattern_score = 2
    elif long_bearish:
        candle_pattern = "nến đỏ thân dài bị đẩy khỏi vùng"
        pattern_score = 2
    elif upper_wick_rejection:
        candle_pattern = "đuôi trên dài/từ chối giá"
        pattern_score = 2
    elif doji:
        candle_pattern = "Doji/thân nhỏ, thị trường do dự"
        pattern_score = 1
    elif latest_touched:
        candle_pattern = "đã chạm nhưng chưa có mẫu nến từ chối rõ"
        pattern_score = 0
    else:
        candle_pattern = "nến mới nhất chưa chạm vùng"
        pattern_score = 0

    rsi14 = float(RSIIndicator(data["close"], window=14).rsi().iloc[-1])
    ema21 = float(EMAIndicator(data["close"], window=21).ema_indicator().iloc[-1])
    ema50 = float(EMAIndicator(data["close"], window=50).ema_indicator().iloc[-1])
    ma_confluence = any(
        lower <= average <= upper
        or (
            float(latest["low"]) <= average <= float(latest["high"])
            and float(latest["close"]) < average
        )
        for average in (ema21, ema50)
    )

    volume_ratio = None
    if "volume" in data.columns:
        volume = pd.to_numeric(data["volume"], errors="coerce")
        baseline = float(volume.iloc[-21:-1].mean())
        latest_volume = float(volume.iloc[-1])
        if pd.notna(baseline) and pd.notna(latest_volume) and baseline > 0:
            volume_ratio = latest_volume / baseline

    rejection_score = 0
    if touched_recently:
        rejection_score += 1
        rejection_score += pattern_score
        if rsi14 >= float(settings.get("rsi_warning", 65)):
            rejection_score += 1
        if ma_confluence:
            rejection_score += 1
        if volume_ratio is not None and volume_ratio >= float(
            settings.get("volume_spike_ratio", 1.5)
        ):
            rejection_score += 1

    if current_price > upper:
        verdict = "NGHI PHÁ KHÁNG CỰ; không short nếu chưa đóng lại dưới vùng"
    elif not touched_recently:
        verdict = "CHƯA TEST; chỉ đặt cảnh báo, chưa có tín hiệu đảo chiều"
    elif rejection_score >= 4:
        verdict = "TỪ CHỐI KHÁ MẠNH; vẫn phải chờ nến 15m đóng xác nhận"
    elif rejection_score >= 2:
        verdict = "CÓ DẤU HIỆU TỪ CHỐI nhưng chưa đủ mạnh"
    else:
        verdict = "ĐANG TEST; chưa có xác nhận từ chối"

    touched_rows = recent[(recent["high"] >= lower) & (recent["low"] <= upper)]
    rejection_low = (
        float(touched_rows["low"].min()) if not touched_rows.empty else lower
    )
    return ResistanceZoneAnalysis(
        lower=lower,
        upper=upper,
        state=state,
        distance=distance,
        touched_recently=touched_recently,
        candle_pattern=candle_pattern,
        rsi14=rsi14,
        ema21=ema21,
        ema50=ema50,
        ma_confluence=ma_confluence,
        volume_ratio=volume_ratio,
        rejection_score=rejection_score,
        verdict=verdict,
        rejection_low=rejection_low,
    )


def build_support_zone(
    current_price: float,
    atr: float,
    structures: dict[str, ChartStructure],
    df: pd.DataFrame,
    settings: dict | None = None,
    resistance_lower: float | None = None,
) -> PriceZone:
    settings = settings or {}
    atr = max(float(atr), current_price * 0.0005)
    cluster_tolerance = atr * float(settings.get("cluster_atr", 0.35))
    padding = max(
        atr * float(settings.get("padding_atr", 0.12)),
        current_price * float(settings.get("minimum_padding_pct", 0.00015)),
    )
    minimum_gap = max(atr * 0.20, current_price * 0.0002)
    ceiling = current_price - padding
    if resistance_lower is not None:
        ceiling = min(ceiling, resistance_lower - minimum_gap)
    levels: list[float] = []
    for structure in structures.values():
        levels.append(float(structure.support))
        levels.extend(float(point.price) for point in structure.swing_lows)
    levels = [
        value
        for value in levels
        if pd.notna(value)
        and current_price - atr * 4 <= value <= ceiling
    ]
    if not levels:
        rolling_lows = [
            float(df["low"].tail(window).min())
            for window in (20, 50)
            if len(df.tail(window))
        ]
        levels = [value for value in rolling_lows if value <= ceiling]
    if not levels:
        levels = [min(ceiling, current_price - atr * 0.75)]
    seed = min(levels, key=lambda value: abs(value - current_price))
    cluster = [
        value for value in levels if abs(value - seed) <= cluster_tolerance
    ] or [seed]
    lower = min(cluster) - padding
    upper = min(max(cluster) + padding, ceiling)
    if upper <= lower:
        lower = upper - max(padding, atr * 0.10)
    return PriceZone(lower, upper)


def assess_breakout_retest(
    df: pd.DataFrame,
    current_price: float,
    atr: float,
    consensus: TimeframeConsensus,
    resistance_test: ResistanceZoneAnalysis,
    structures: dict[str, ChartStructure],
    legacy_long_confirmed: bool,
    legacy_short_confirmed: bool,
    settings: dict | None = None,
) -> RetestAssessment:
    """Require a completed 15m breakout/breakdown followed by a completed retest."""
    settings = settings or {}
    lookback = max(3, int(settings.get("retest_lookback_bars", 8)))
    tolerance = max(
        atr * float(settings.get("retest_tolerance_atr", 0.15)),
        current_price * 0.0001,
    )
    support = build_support_zone(
        current_price,
        atr,
        structures,
        df,
        settings,
        resistance_lower=resistance_test.lower,
    )
    resistance = PriceZone(resistance_test.lower, resistance_test.upper)
    recent = df.tail(lookback).copy()

    long_breakout_at = None
    short_breakdown_at = None
    for position in range(1, len(recent)):
        previous_close = float(recent["close"].iloc[position - 1])
        close = float(recent["close"].iloc[position])
        if (
            previous_close <= resistance.upper
            and close > resistance.upper
        ):
            long_breakout_at = position
        if (
            previous_close >= support.lower
            and close < support.lower
        ):
            short_breakdown_at = position

    long_retest_confirmed = False
    if long_breakout_at is not None and long_breakout_at + 1 < len(recent):
        after_breakout = recent.iloc[long_breakout_at + 1 :]
        long_retest_confirmed = bool(
            (
                (after_breakout["low"] <= resistance.upper + tolerance)
                & (after_breakout["low"] >= resistance.lower - tolerance)
                & (after_breakout["close"] > resistance.upper)
            ).any()
        )

    short_retest_confirmed = False
    if short_breakdown_at is not None and short_breakdown_at + 1 < len(recent):
        after_breakdown = recent.iloc[short_breakdown_at + 1 :]
        short_retest_confirmed = bool(
            (
                (after_breakdown["high"] >= support.lower - tolerance)
                & (after_breakdown["high"] <= support.upper + tolerance)
                & (after_breakdown["close"] < support.lower)
            ).any()
        )

    if long_retest_confirmed:
        long_phase = "đã phá R và retest giữ được"
    elif long_breakout_at is not None:
        long_phase = "đã đóng trên R, đang chờ retest"
    else:
        long_phase = f"chờ nến 15m đóng trên {resistance.upper:.2f}"

    if short_retest_confirmed:
        short_phase = "đã thủng S và retest không vượt lại"
    elif short_breakdown_at is not None:
        short_phase = "đã đóng dưới S, đang chờ retest"
    else:
        short_phase = f"chờ nến 15m đóng dưới {support.lower:.2f}"

    actionable_side = None
    decision_reason = "CHỜ: chưa đủ chuỗi nến đóng phá vùng và retest"
    entry_lower = None
    entry_upper = None
    invalidation = None
    long_entry_lower = resistance.upper
    long_entry_upper = resistance.upper + tolerance
    short_entry_lower = support.lower - tolerance
    short_entry_upper = support.lower
    long_price_in_entry = long_entry_lower <= current_price <= long_entry_upper
    short_price_in_entry = short_entry_lower <= current_price <= short_entry_upper
    if (
        long_retest_confirmed
        and consensus.actionable
        and consensus.score > 0
        and legacy_long_confirmed
        and long_price_in_entry
    ):
        actionable_side = "LONG"
        decision_reason = "LONG: đa khung đồng thuận, breakout và retest đã xác nhận"
        entry_lower = long_entry_lower
        entry_upper = long_entry_upper
        invalidation = resistance.lower - tolerance
    elif (
        short_retest_confirmed
        and consensus.actionable
        and consensus.score < 0
        and legacy_short_confirmed
        and short_price_in_entry
    ):
        actionable_side = "SHORT"
        decision_reason = "SHORT: đa khung đồng thuận, breakdown và retest đã xác nhận"
        entry_lower = short_entry_lower
        entry_upper = short_entry_upper
        invalidation = support.upper + tolerance
    elif (
        long_retest_confirmed
        and consensus.actionable
        and consensus.score > 0
        and legacy_long_confirmed
    ):
        decision_reason = "CHỜ: LONG đã xác nhận nhưng giá hiện ngoài vùng vào; đợi retest lại"
    elif (
        short_retest_confirmed
        and consensus.actionable
        and consensus.score < 0
        and legacy_short_confirmed
    ):
        decision_reason = "CHỜ: SHORT đã xác nhận nhưng giá hiện ngoài vùng vào; đợi retest lại"
    elif consensus.actionable and (
        long_retest_confirmed or short_retest_confirmed
    ):
        decision_reason = "CHỜ: retest có nhưng tín hiệu backtest gần đây chưa xác nhận"
    elif not consensus.actionable:
        decision_reason = "CHỜ: ba khung nến đã đóng chưa đồng thuận đủ mạnh"

    return RetestAssessment(
        support=support,
        resistance=resistance,
        long_phase=long_phase,
        short_phase=short_phase,
        long_retest_confirmed=long_retest_confirmed,
        short_retest_confirmed=short_retest_confirmed,
        actionable_side=actionable_side,
        decision_reason=decision_reason,
        entry_lower=entry_lower,
        entry_upper=entry_upper,
        invalidation=invalidation,
    )


def render_price_chart(
    df: pd.DataFrame,
    plan=None,
    structure: ChartStructure | None = None,
    resistance_test: ResistanceZoneAnalysis | None = None,
    retest_assessment: RetestAssessment | None = None,
    interval_label: str = "15m",
    current_price: float | None = None,
) -> BytesIO:
    data = df.tail(60).copy()
    ema9 = EMAIndicator(data["close"], window=9).ema_indicator()
    ema21 = EMAIndicator(data["close"], window=21).ema_indicator()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#10151d")
    ax.set_facecolor("#10151d")
    width = 0.62
    for x, (_, row) in enumerate(data.iterrows()):
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax.vlines(x, row["low"], row["high"], color=color, linewidth=1)
        bottom = min(row["open"], row["close"])
        height = max(abs(row["close"] - row["open"]), 0.02)
        ax.add_patch(plt.Rectangle((x - width / 2, bottom), width, height, color=color, alpha=0.9))

    ax.plot(range(len(data)), ema9, color="#ffd166", linewidth=1.3, label="EMA 9")
    ax.plot(range(len(data)), ema21, color="#4cc9f0", linewidth=1.3, label="EMA 21")
    if structure is not None:
        position_by_time = {timestamp: index for index, timestamp in enumerate(data.index)}
        for point in structure.swing_highs:
            if point.timestamp in position_by_time:
                x = position_by_time[point.timestamp]
                ax.scatter(x, point.price, marker="v", color="#ffb703", s=45, zorder=5)
                ax.annotate("H", (x, point.price), color="#ffb703", xytext=(0, 7), textcoords="offset points")
        for point in structure.swing_lows:
            if point.timestamp in position_by_time:
                x = position_by_time[point.timestamp]
                ax.scatter(x, point.price, marker="^", color="#90be6d", s=45, zorder=5)
                ax.annotate("L", (x, point.price), color="#90be6d", xytext=(0, -14), textcoords="offset points")
        ax.axhline(structure.support, color="#90be6d", linestyle=":", linewidth=1, label="Support")
        ax.axhline(structure.resistance, color="#ffb703", linestyle=":", linewidth=1, label="Resistance")
    if resistance_test is not None:
        ax.axhspan(
            resistance_test.lower,
            resistance_test.upper,
            color="#ff8c42",
            alpha=0.14,
            label="Resistance zone",
        )
    if retest_assessment is not None:
        ax.axhspan(
            retest_assessment.support.lower,
            retest_assessment.support.upper,
            color="#43aa8b",
            alpha=0.13,
            label="Support zone",
        )
    if current_price is not None:
        ax.axhline(
            current_price,
            color="#f8f9fa",
            linestyle="-.",
            linewidth=0.9,
            alpha=0.8,
            label="Live price",
        )
    if plan is not None:
        ax.axhline(plan.entry, color="#ffffff", linestyle="--", linewidth=1, label="Entry")
        ax.axhline(plan.stop, color="#ef5350", linestyle="--", linewidth=1, label="SL")
        ax.axhline(plan.take_profit_2, color="#26a69a", linestyle="--", linewidth=1, label="TP2")

    tick_count = min(6, len(data))
    ticks = [round(i * (len(data) - 1) / max(1, tick_count - 1)) for i in range(tick_count)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([data.index[i].strftime("%d/%m %H:%M") for i in ticks], color="#aab2bf")
    ax.tick_params(axis="y", colors="#aab2bf")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(alpha=0.12)
    ax.set_title(f"XAU/USD - {interval_label} (60 nến đã đóng)", color="white")
    ax.legend(loc="upper left", facecolor="#18212f", labelcolor="white")
    fig.tight_layout()

    output = BytesIO()
    output.name = "xau_15m.png"
    fig.savefig(output, format="png", dpi=135, facecolor=fig.get_facecolor())
    plt.close(fig)
    output.seek(0)
    return output


def render_multi_timeframe_chart(
    frames: dict[str, pd.DataFrame],
    structures: dict[str, ChartStructure],
    resistance_test: ResistanceZoneAnalysis,
    retest_assessment: RetestAssessment,
    current_price: float,
    plan=None,
) -> BytesIO:
    """Render one compact image containing the closed 15m, 1h, and 4h charts."""
    ordered = [tf for tf in ("15min", "1h", "4h") if tf in frames]
    fig, axes = plt.subplots(len(ordered), 1, figsize=(10, 9))
    if len(ordered) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#10151d")

    for ax, timeframe in zip(axes, ordered):
        data = frames[timeframe].tail(50).copy()
        ax.set_facecolor("#10151d")
        width = 0.62
        for x, (_, row) in enumerate(data.iterrows()):
            color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
            ax.vlines(x, row["low"], row["high"], color=color, linewidth=0.8)
            bottom = min(row["open"], row["close"])
            height = max(abs(row["close"] - row["open"]), 0.02)
            ax.add_patch(
                plt.Rectangle(
                    (x - width / 2, bottom),
                    width,
                    height,
                    color=color,
                    alpha=0.9,
                )
            )

        ema9 = EMAIndicator(data["close"], window=9).ema_indicator()
        ema21 = EMAIndicator(data["close"], window=21).ema_indicator()
        ax.plot(range(len(data)), ema9, color="#ffd166", linewidth=1, label="EMA9")
        ax.plot(range(len(data)), ema21, color="#4cc9f0", linewidth=1, label="EMA21")
        ax.axhspan(
            retest_assessment.support.lower,
            retest_assessment.support.upper,
            color="#43aa8b",
            alpha=0.12,
            label="S",
        )
        ax.axhspan(
            resistance_test.lower,
            resistance_test.upper,
            color="#ff8c42",
            alpha=0.13,
            label="R",
        )
        ax.axhline(
            current_price,
            color="#f8f9fa",
            linestyle="-.",
            linewidth=0.8,
            alpha=0.8,
            label="Live",
        )
        if plan is not None:
            ax.axhline(plan.entry, color="#ffffff", linestyle="--", linewidth=0.8)
            ax.axhline(plan.stop, color="#ef5350", linestyle="--", linewidth=0.8)
            ax.axhline(plan.take_profit_2, color="#26a69a", linestyle="--", linewidth=0.8)

        tick_count = min(5, len(data))
        ticks = [
            round(i * (len(data) - 1) / max(1, tick_count - 1))
            for i in range(tick_count)
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [
                data.index[i].tz_convert("Asia/Bangkok").strftime("%d/%m %H:%M")
                for i in ticks
            ],
            color="#aab2bf",
            fontsize=7,
        )
        ax.tick_params(axis="y", colors="#aab2bf", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(alpha=0.1)
        structure = structures[timeframe]
        ax.set_title(
            f"{timeframe} · {structure.pattern} · nến đã đóng",
            color="white",
            fontsize=10,
            loc="left",
        )
        ax.legend(
            loc="upper left",
            ncol=5,
            fontsize=7,
            facecolor="#18212f",
            labelcolor="white",
        )

    fig.suptitle("XAU/USD · 15m / 1H / 4H", color="white", fontsize=13)
    fig.tight_layout()
    output = BytesIO()
    output.name = "xau_multi_timeframe.png"
    fig.savefig(output, format="png", dpi=95, facecolor=fig.get_facecolor())
    plt.close(fig)
    output.seek(0)
    return output
