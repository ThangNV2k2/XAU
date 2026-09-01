"""Deterministic multi-timeframe RSI/trend/retest strategy.

This engine produces a plan only after a closed-candle retest.  Its confidence
is setup completeness, not an advertised win probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Mapping

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange

from market_sessions import NEW_YORK, SessionState


MIN_BARS = 80

# EMA200 is recursive and seeded from the first bar of whatever slice it gets,
# so a short window leaves real weight on one arbitrary old price: at 260 bars
# ~7.4% of the value is still the seed, which moved EMA200 by ~4 USD on gold and
# flipped the close-vs-EMA200 side on ~2% of bars between the live slice and the
# full-history replay. Below ~800 bars the residual is under 0.05%.
EMA200_WARMUP_BARS = 800

# ATR-normalised components are divided by these multiples of ATR before being
# clipped. Dividing by a bare 1x ATR pinned (ema20-ema50)/atr at +/-1 on 54% of
# bars and (close-ema20)/atr on 61% (measured on 20k bars), which turned two of
# the five components into sign flags and made their weights meaningless. The
# divisors below are the p90 of |component| so the clip is a tail cut, not a
# median cut; MACD needed the opposite treatment because it never reached +/-1.
EMA_SPREAD_ATR_DIVISOR = 2.5
PRICE_DISTANCE_ATR_DIVISOR = 3.0
MACD_ATR_DIVISOR = 0.5


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(frozen=True)
class TechnicalSnapshot:
    interval: str
    close: float
    rsi: float
    rsi_slope: float
    ema20: float
    ema50: float
    ema200: float | None
    macd_histogram: float
    adx: float
    atr: float
    score: float


@dataclass(frozen=True)
class RetestSignal:
    confirmed: bool
    kind: str
    level: float | None
    candle_low: float | None
    candle_high: float | None
    reason: str


@dataclass(frozen=True)
class TradePlan:
    side: str
    setup: str
    entry_lower: float
    entry_upper: float
    preferred_entry: float
    invalidation_level: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk: float
    reward_risk_1: float
    reward_risk_2: float


@dataclass(frozen=True)
class MarketAnalysis:
    symbol: str
    asset_type: str
    checked_at: datetime
    market_phase: str
    action: str
    confidence: int
    confidence_note: str
    intraday_score: float
    long_term_score: float
    bias: str
    bias_reason: str
    horizon: str
    reason: str
    snapshots: Mapping[str, TechnicalSnapshot]
    retest: RetestSignal
    plan: TradePlan | None


def technical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate the live strategy indicators for every closed candle.

    Historical replay uses this vectorized result, so it cannot quietly drift
    away from the formula used by the live scanner. Sharing the formula is not
    enough on its own: EMA200 also needs the same warm-up on both sides, which
    is why it is dropped below ``EMA200_WARMUP_BARS`` rather than computed from
    a short slice that would only agree with the replay by accident.
    """
    if len(frame) < MIN_BARS:
        raise ValueError(f"Cần ít nhất {MIN_BARS} nến đóng, hiện có {len(frame)}")
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    rsi_series = RSIIndicator(close, window=14).rsi()
    ema20_series = EMAIndicator(close, window=20).ema_indicator()
    ema50_series = EMAIndicator(close, window=50).ema_indicator()
    ema200_series = (
        EMAIndicator(close, window=200).ema_indicator()
        if len(frame) >= EMA200_WARMUP_BARS
        else None
    )
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_histogram = macd.macd_diff()
    atr_series = AverageTrueRange(high, low, close, window=14).average_true_range()
    adx_series = ADXIndicator(high, low, close, window=14).adx()

    # RSI is used as a trend-regime measure around 50, not as a naive
    # "oversold means buy" reversal rule.
    atr_safe = atr_series.clip(lower=1e-9)
    rsi_slope = rsi_series - rsi_series.shift(3)
    rsi_component = ((rsi_series - 50.0) / 18.0).clip(-1.0, 1.0)
    rsi_slope_component = (rsi_slope / 8.0).clip(-1.0, 1.0)
    ema_component = (
        (ema20_series - ema50_series) / (EMA_SPREAD_ATR_DIVISOR * atr_safe)
    ).clip(-1.0, 1.0)
    price_component = (
        (close - ema20_series) / (PRICE_DISTANCE_ATR_DIVISOR * atr_safe)
    ).clip(-1.0, 1.0)
    macd_component = (macd_histogram / (MACD_ATR_DIVISOR * atr_safe)).clip(-1.0, 1.0)
    score = (
        0.28 * rsi_component
        + 0.12 * rsi_slope_component
        + 0.30 * ema_component
        + 0.15 * price_component
        + 0.15 * macd_component
    )
    if ema200_series is not None:
        long_trend = close.gt(ema200_series).astype(float).mul(2.0).sub(1.0)
        score = score.where(ema200_series.isna(), 0.90 * score + 0.10 * long_trend)
    # Low ADX shrinks conviction without reversing direction.
    trend_factor = 0.70 + 0.30 * (adx_series / 25.0).clip(0.0, 1.0)
    score = (score * trend_factor).clip(-1.0, 1.0)
    return pd.DataFrame(
        {
            "close": close,
            "rsi": rsi_series,
            "rsi_slope": rsi_slope,
            "ema20": ema20_series,
            "ema50": ema50_series,
            "ema200": ema200_series if ema200_series is not None else float("nan"),
            "macd_histogram": macd_histogram,
            "adx": adx_series,
            "atr": atr_safe,
            "score": score,
        },
        index=frame.index,
    )


def technical_snapshot(frame: pd.DataFrame, interval: str) -> TechnicalSnapshot:
    values = technical_frame(frame).iloc[-1]
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


def opening_range_levels(
    frame_1m: pd.DataFrame,
    now: datetime,
    asset_type: str,
) -> tuple[float, float] | None:
    local_now = now.astimezone(NEW_YORK)
    local_index = frame_1m.index
    if local_index.tz is None:
        local_index = local_index.tz_localize(timezone.utc)
    local_index = local_index.tz_convert(NEW_YORK)
    if asset_type == "stock":
        start, end = time(9, 40), time(9, 55)
    else:
        start, end = time(9, 30), time(9, 45)
    mask = (
        (local_index.date == local_now.date())
        & (local_index.time >= start)
        & (local_index.time < end)
    )
    opening = frame_1m.loc[mask]
    if opening.empty:
        return None
    return float(opening["high"].max()), float(opening["low"].min())


def _rejection(candle: pd.Series, side: str, minimum_close_location: float) -> bool:
    high, low = float(candle["high"]), float(candle["low"])
    close = float(candle["close"])
    span = high - low
    if span <= 0.0:
        # A zero-range bar carries no rejection information. Without this guard
        # close_location collapses to 0.0, which silently reads as a perfect
        # bearish rejection and lets a frozen feed confirm a SHORT retest.
        return False
    close_location = (close - low) / span
    return close_location >= minimum_close_location if side == "LONG" else close_location <= 1 - minimum_close_location


def detect_retest(
    frame: pd.DataFrame,
    snapshot: TechnicalSnapshot,
    side: str,
    settings: dict,
    focus_level: float | None = None,
) -> RetestSignal:
    lookback = max(3, int(settings.get("retest_lookback_bars", 8)))
    tolerance = snapshot.atr * float(settings.get("retest_tolerance_atr", 0.18))
    minimum_close_location = float(settings.get("minimum_rejection_close_location", 0.60))
    recent = frame.tail(lookback + 22)
    if len(recent) < 5:
        return RetestSignal(False, "NONE", None, None, None, "Chưa đủ nến tìm retest")

    if focus_level is not None:
        level = float(focus_level)
        breakout_positions = []
        for position in range(max(0, len(recent) - lookback - 1), len(recent) - 1):
            close = float(recent.iloc[position]["close"])
            if (side == "LONG" and close > level) or (side == "SHORT" and close < level):
                breakout_positions.append(position)
        for breakout in reversed(breakout_positions):
            for position in range(breakout + 1, len(recent)):
                candle = recent.iloc[position]
                touches = (
                    float(candle["low"]) <= level + tolerance
                    and float(candle["close"]) > level
                    if side == "LONG"
                    else float(candle["high"]) >= level - tolerance
                    and float(candle["close"]) < level
                )
                if touches and _rejection(candle, side, minimum_close_location):
                    # Confirmation must be fresh, otherwise the entry is stale.
                    if position < len(recent) - 2:
                        continue
                    return RetestSignal(
                        True,
                        "US_OPENING_RANGE_RETEST",
                        level,
                        float(candle["low"]),
                        float(candle["high"]),
                        "Đã breakout và retest biên mở phiên bằng nến đóng",
                    )
        return RetestSignal(
            False,
            "US_OPENING_RANGE_RETEST",
            level,
            None,
            None,
            "Chờ nến đóng breakout rồi retest biên mở phiên",
        )

    highs = recent["high"].astype(float)
    lows = recent["low"].astype(float)
    closes = recent["close"].astype(float)
    resistance = highs.shift(1).rolling(20).max()
    support = lows.shift(1).rolling(20).min()
    start = max(1, len(recent) - lookback - 1)
    for breakout in range(len(recent) - 2, start - 1, -1):
        level_value = resistance.iloc[breakout] if side == "LONG" else support.iloc[breakout]
        if pd.isna(level_value):
            continue
        broke = closes.iloc[breakout] > level_value if side == "LONG" else closes.iloc[breakout] < level_value
        if not broke:
            continue
        level = float(level_value)
        for position in range(breakout + 1, len(recent)):
            candle = recent.iloc[position]
            touches = (
                float(candle["low"]) <= level + tolerance and float(candle["close"]) > level
                if side == "LONG"
                else float(candle["high"]) >= level - tolerance and float(candle["close"]) < level
            )
            if touches and _rejection(candle, side, minimum_close_location) and position >= len(recent) - 2:
                return RetestSignal(
                    True,
                    "BREAKOUT_RETEST",
                    level,
                    float(candle["low"]),
                    float(candle["high"]),
                    "Breakout cấu trúc đã có nến đóng retest xác nhận",
                )

    # Secondary standard setup: a pullback to EMA20 while the 20/50 trend is
    # intact. This remains a retest and still requires a rejection candle.
    latest = recent.iloc[-1]
    level = snapshot.ema20
    trend_ok = snapshot.ema20 > snapshot.ema50 if side == "LONG" else snapshot.ema20 < snapshot.ema50
    touches = (
        float(latest["low"]) <= level + tolerance and float(latest["close"]) > level
        if side == "LONG"
        else float(latest["high"]) >= level - tolerance and float(latest["close"]) < level
    )
    if trend_ok and touches and _rejection(latest, side, minimum_close_location):
        return RetestSignal(
            True,
            "EMA20_RETEST",
            level,
            float(latest["low"]),
            float(latest["high"]),
            "Xu hướng EMA20/50 còn nguyên và nến đóng từ chối EMA20",
        )
    return RetestSignal(False, "TREND_RETEST", level, None, None, "Chờ giá retest cấu trúc hoặc EMA20")


def higher_timeframe_bias(
    snapshots: Mapping[str, TechnicalSnapshot],
    settings: Mapping[str, object] | None = None,
) -> tuple[str, str]:
    """Thiên hướng của khung lớn: LONG, SHORT hoặc NEUTRAL.

    Đây là bộ lọc cứng, không phải một số hạng có trọng số. Trên 15 tháng nến
    Exness thật, chiến lược mở 150 lệnh SHORT trong một sóng tăng 70% và lỗ
    34,75R vì xu hướng khung lớn chỉ chiếm 10% trọng số trong ``score`` — quá
    nhẹ để chặn bất cứ thứ gì. NEUTRAL cũng chặn lệnh: khi khung ngày chưa ngã
    ngũ thì không có "chắc chắn" nào để vào.
    """
    settings = settings or {}
    interval = str(settings.get("bias_interval", "1day"))
    fallback = str(settings.get("bias_fallback_interval", "4h"))
    snapshot = snapshots.get(interval) or snapshots.get(fallback)
    if snapshot is None:
        return "NEUTRAL", "Thiếu khung xác định thiên hướng"
    label = snapshot.interval
    # Cố tình KHÔNG dùng EMA200 ở đây, dù snapshot có sẵn. Live nạp 1000 nến D1
    # nên EMA200 có; một backtest 15 tháng chỉ có ~390 nến D1 nên không có. Nếu
    # luật rẽ nhánh theo đó thì live chạy một luật còn backtest kiểm luật khác —
    # đúng loại phân kỳ mà EMA200_WARMUP_BARS sinh ra để ngăn. Cấu trúc
    # EMA20/EMA50 + vị trí giá cần ~50 nến, giống hệt nhau ở cả hai đường.
    if snapshot.ema20 > snapshot.ema50 and snapshot.close > snapshot.ema50:
        return "LONG", f"{label} EMA20>EMA50 và giá trên EMA50"
    if snapshot.ema20 < snapshot.ema50 and snapshot.close < snapshot.ema50:
        return "SHORT", f"{label} EMA20<EMA50 và giá dưới EMA50"
    return "NEUTRAL", f"{label} chưa ngã ngũ quanh EMA50"


def _weighted_score(snapshots: Mapping[str, TechnicalSnapshot], weights: Mapping[str, float]) -> float:
    available = [(snapshots[key].score, value) for key, value in weights.items() if key in snapshots]
    total_weight = sum(weight for _, weight in available)
    if total_weight <= 0:
        return 0.0
    return _clip(sum(score * weight for score, weight in available) / total_weight)


def _round(value: float, digits: int) -> float:
    return round(float(value), max(0, min(8, digits)))


def analyze_market(
    symbol: str,
    asset_type: str,
    frames: Mapping[str, pd.DataFrame],
    quote: Mapping[str, object],
    session: SessionState,
    settings: dict | None = None,
    now: datetime | None = None,
    precomputed_snapshots: Mapping[str, TechnicalSnapshot] | None = None,
) -> MarketAnalysis:
    settings = settings or {}
    checked_at = now or datetime.now(timezone.utc)
    if checked_at.tzinfo is None:
        checked_at = checked_at.replace(tzinfo=timezone.utc)
    snapshots = dict(precomputed_snapshots) if precomputed_snapshots is not None else {
        key: technical_snapshot(frame, key) for key, frame in frames.items()
    }
    intraday_weights = settings.get(
        "intraday_weights",
        {"1min": 0.05, "5min": 0.20, "15min": 0.30, "1h": 0.30, "4h": 0.15},
    )
    long_term_weights = settings.get(
        "long_term_weights",
        {"1h": 0.20, "4h": 0.50, "1day": 0.30},
    )
    intraday_score = _weighted_score(snapshots, intraday_weights)
    long_term_score = _weighted_score(snapshots, long_term_weights)
    threshold = float(settings.get("actionable_score", 0.30))
    alignment_floor = float(settings.get("alignment_floor", 0.08))
    higher = [snapshots[key].score for key in ("15min", "1h", "4h") if key in snapshots]
    long_aligned = len(higher) >= 2 and sum(score >= alignment_floor for score in higher) >= 2
    short_aligned = len(higher) >= 2 and sum(score <= -alignment_floor for score in higher) >= 2

    bias, bias_reason = higher_timeframe_bias(snapshots, settings)
    require_bias = bool(settings.get("require_bias_alignment", True))

    side = None
    if intraday_score >= threshold and long_aligned:
        side = "LONG"
    elif intraday_score <= -threshold and short_aligned:
        side = "SHORT"
    # Ngược thiên hướng khung ngày thì bỏ hẳn, không hạ điểm tin cậy rồi vẫn vào.
    blocked_by_bias = require_bias and side is not None and side != bias
    rejected_side = side if blocked_by_bias else None
    if blocked_by_bias:
        side = None

    entry_interval = str(settings.get("entry_interval", "5min"))
    entry_snapshot = snapshots.get(entry_interval) or snapshots.get("15min")
    if entry_snapshot is None:
        raise ValueError("Thiếu khung vào lệnh 5min/15min")
    rsi_long = settings.get("rsi_long_zone", [45, 68])
    rsi_short = settings.get("rsi_short_zone", [32, 55])
    rsi_ok = bool(
        side == "LONG" and float(rsi_long[0]) <= entry_snapshot.rsi <= float(rsi_long[1]) and entry_snapshot.rsi_slope >= 0
        or side == "SHORT" and float(rsi_short[0]) <= entry_snapshot.rsi <= float(rsi_short[1]) and entry_snapshot.rsi_slope <= 0
    )

    # The normal multi-timeframe/retest model always runs, including during the
    # US session. Opening-range handling is an additional setup detector; it
    # must never replace RSI/EMA/MACD/ADX/ATR or the standard retest path.
    standard_retest = (
        detect_retest(frames[entry_interval], entry_snapshot, side, settings)
        if side is not None and entry_interval in frames
        else RetestSignal(False, "NONE", None, None, None, "Các khung chưa đồng thuận")
    )
    opening_retest = None
    opening_phase = session.phase in {"RETEST_WINDOW", "US_RETEST_WINDOW"}
    opening = opening_range_levels(frames["1min"], checked_at, asset_type) if opening_phase and "1min" in frames else None
    if side and opening is not None:
        focus_level = opening[0] if side == "LONG" else opening[1]
        opening_retest = detect_retest(
            frames[entry_interval],
            entry_snapshot,
            side,
            settings,
            focus_level,
        )
    if opening_retest is not None and opening_retest.confirmed:
        retest = opening_retest
    elif standard_retest.confirmed:
        retest = standard_retest
    elif opening_retest is not None:
        retest = RetestSignal(
            False,
            "US_AND_STANDARD_RETEST",
            standard_retest.level,
            None,
            None,
            f"{standard_retest.reason}; lớp phiên Mỹ: {opening_retest.reason}",
        )
    else:
        retest = standard_retest

    blockers: list[str] = []
    if not session.is_open:
        blockers.append(session.reason)
    elif not session.allow_new_entry:
        blockers.append(session.reason)
    if rejected_side is not None:
        blockers.append(
            f"{rejected_side} ngược thiên hướng khung lớn ({bias}: {bias_reason}); không đánh ngược sóng"
        )
    elif side is None:
        blockers.append("15m/1H/4H chưa đồng thuận đủ mạnh")
    if side is not None and not rsi_ok:
        blockers.append(f"RSI {entry_interval}={entry_snapshot.rsi:.1f} chưa ở vùng retest an toàn")
    if side is not None and not retest.confirmed:
        blockers.append(retest.reason)

    plan = None
    if side and retest.confirmed and retest.level is not None:
        # Derived once and reused by the plan below; computing the entry band
        # twice let a blocker and the plan it guards drift apart silently.
        atr = entry_snapshot.atr
        level = retest.level
        tolerance = atr * float(settings.get("entry_tolerance_atr", 0.10))
        lower, upper = level - tolerance, level + tolerance
        market_entry = quote.get("ask") if side == "LONG" else quote.get("bid")
        if market_entry is None:
            market_entry = quote.get("close", level)
        preferred = float(market_entry)
        # ATR only guards against divide-by-zero, so a frozen or holiday feed
        # can still score above threshold on a single tick and hand back a plan
        # whose whole stop is a few cents wide. Require real volatility first.
        minimum_atr_ratio = float(settings.get("minimum_atr_ratio", 0.0002))
        reference_price = abs(preferred) or abs(level)
        if reference_price > 0 and atr / reference_price < minimum_atr_ratio:
            blockers.append(
                f"Biến động quá thấp (ATR {atr:g} = {atr / reference_price * 100:.4f}% giá); "
                "nhiều khả năng feed đứng hoặc nghỉ lễ"
            )
        if not lower <= preferred <= upper:
            blockers.append(
                f"Giá khớp Exness {preferred:g} đã ra ngoài vùng retest {lower:g}–{upper:g}; không đuổi giá"
            )
        bid, ask = quote.get("bid"), quote.get("ask")
        if bid is not None and ask is not None:
            spread_atr = (float(ask) - float(bid)) / atr
            if spread_atr > float(settings.get("maximum_spread_atr", 0.12)):
                blockers.append(f"Spread Exness quá rộng ({spread_atr:.2f} ATR)")

        # Nested so the plan can only ever be built from the entry band and
        # blockers computed just above, for the same retest.
        if not blockers:
            stop_buffer = atr * float(settings.get("stop_buffer_atr", 0.18))
            minimum_stop = atr * float(settings.get("minimum_stop_atr", 0.65))
            if side == "LONG":
                structural = min(float(retest.candle_low or level), level - minimum_stop)
                stop = structural - stop_buffer
                risk = preferred - stop
            else:
                structural = max(float(retest.candle_high or level), level + minimum_stop)
                stop = structural + stop_buffer
                risk = stop - preferred
            maximum_stop_atr = float(settings.get("maximum_stop_atr", 2.0))
            if risk <= 0 or risk > maximum_stop_atr * atr:
                blockers.append("SL cấu trúc quá xa; bỏ setup thay vì kéo SL vào gần")
            else:
                tp1_r = float(settings.get("take_profit_1_r", 1.5))
                tp2_r = float(settings.get("take_profit_2_r", 2.5))
                direction = 1 if side == "LONG" else -1
                digits = int(quote.get("digits", 2) or 2)
                plan = TradePlan(
                    side=side,
                    setup=retest.kind,
                    entry_lower=_round(lower, digits),
                    entry_upper=_round(upper, digits),
                    preferred_entry=_round(preferred, digits),
                    invalidation_level=_round(
                        level
                        + (-1 if side == "LONG" else 1)
                        * atr
                        * float(settings.get("invalidation_buffer_atr", 0.12)),
                        digits,
                    ),
                    stop_loss=_round(stop, digits),
                    take_profit_1=_round(preferred + direction * risk * tp1_r, digits),
                    take_profit_2=_round(preferred + direction * risk * tp2_r, digits),
                    risk=_round(risk, digits),
                    reward_risk_1=tp1_r,
                    reward_risk_2=tp2_r,
                )

    action = "WAIT" if plan is None else ("BUY" if plan.side == "LONG" else "SHORT")
    # Count only the higher timeframes that agree with the side actually taken.
    # Taking max() over both directions credited a SHORT setup for the bullish
    # frames that were arguing against it.
    if side == "SHORT":
        aligned_count = sum(score <= -alignment_floor for score in higher)
    elif side == "LONG":
        aligned_count = sum(score >= alignment_floor for score in higher)
    else:
        aligned_count = 0
    confidence = int(
        round(
            min(
                95.0,
                35.0
                + abs(intraday_score) * 30.0
                + aligned_count * 7.0
                + (12.0 if retest.confirmed else 0.0)
                + min(8.0, entry_snapshot.adx / 5.0),
            )
        )
    )
    if long_term_score >= 0.22:
        horizon = "4H–D1 nghiêng tăng"
    elif long_term_score <= -0.22:
        horizon = "4H–D1 nghiêng giảm"
    else:
        horizon = "4H–D1 trung tính/đi ngang"
    reason = "; ".join(blockers) if blockers else retest.reason
    return MarketAnalysis(
        symbol=symbol,
        asset_type=asset_type,
        checked_at=checked_at,
        market_phase=session.phase,
        action=action,
        confidence=confidence,
        confidence_note="độ đầy đủ setup, không phải xác suất thắng",
        intraday_score=intraday_score,
        long_term_score=long_term_score,
        bias=bias,
        bias_reason=bias_reason,
        horizon=horizon,
        reason=reason,
        snapshots=snapshots,
        retest=retest,
        plan=plan,
    )


def format_analysis(analysis: MarketAnalysis, compact: bool = False) -> str:
    icon = {"BUY": "🟢", "SHORT": "🔴", "WAIT": "⚪"}[analysis.action]
    headline = f"{icon} {analysis.symbol} · {analysis.action} · {analysis.market_phase}"
    score_line = (
        f"Intraday {analysis.intraday_score:+.2f} · dài hạn {analysis.long_term_score:+.2f} · "
        f"setup {analysis.confidence}/100 ({analysis.confidence_note})"
    )
    if compact:
        if analysis.plan:
            plan = analysis.plan
            return (
                f"{headline}\nEntry {plan.entry_lower:g}–{plan.entry_upper:g} (ưu tiên {plan.preferred_entry:g}) · "
                f"Hủy {plan.invalidation_level:g} · SL {plan.stop_loss:g} · "
                f"TP1 {plan.take_profit_1:g} · TP2 {plan.take_profit_2:g}\n{score_line}"
            )
        return f"{headline}\n{analysis.horizon} · {analysis.reason}\n{score_line}"
    bias_icon = {"LONG": "▲", "SHORT": "▼", "NEUTRAL": "◆"}[analysis.bias]
    lines = [
        headline,
        score_line,
        f"Thiên hướng khung lớn: {bias_icon} {analysis.bias} ({analysis.bias_reason})",
        f"Dự báo: {analysis.horizon}",
    ]
    for interval in ("1min", "5min", "15min", "1h", "4h", "1day"):
        snap = analysis.snapshots.get(interval)
        if snap:
            lines.append(
                f"• {interval}: RSI {snap.rsi:.1f} ({snap.rsi_slope:+.1f}) · "
                f"ADX {snap.adx:.1f} · score {snap.score:+.2f}"
            )
    if analysis.plan:
        plan = analysis.plan
        lines.extend(
            [
                f"Setup: {plan.setup}",
                f"Entry: {plan.entry_lower:g}–{plan.entry_upper:g} · ưu tiên {plan.preferred_entry:g}",
                f"Mức hủy setup khi nến đóng xác nhận: {plan.invalidation_level:g}",
                f"SL: {plan.stop_loss:g} · TP1: {plan.take_profit_1:g} ({plan.reward_risk_1:g}R) · "
                f"TP2: {plan.take_profit_2:g} ({plan.reward_risk_2:g}R)",
                "Quản trị: chốt một phần ở TP1, dời SL về hòa vốn; không đuổi giá ngoài vùng Entry.",
            ]
        )
    else:
        lines.append(f"Chưa vào lệnh: {analysis.reason}")
    lines.append("Chỉ là tín hiệu định lượng từ dữ liệu Exness, cần paper-test trước khi dùng tiền thật.")
    return "\n".join(lines)
