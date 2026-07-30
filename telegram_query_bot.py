import asyncio
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from analysis_diagnostics import (
    append_analysis_snapshot,
    build_analysis_snapshot,
    build_diagnostic_event,
)

from data_provider.binance_futures_provider import BinanceFuturesProvider
from indicators.signal_engine import (
    MomentumBias,
    SignalResult,
    compute_momentum_bias,
    compute_signal,
    compute_signal_series,
)
from ai_analysis import (
    ai_daily_budget_available,
    analyze_with_gemini,
    analyze_with_groq,
    analyze_peak_with_gemini,
    analyze_peak_with_groq,
    build_ai_snapshot,
    build_peak_ai_snapshot,
    format_ai_analysis,
    format_peak_ai_review,
    is_ai_rate_limit_error,
    record_ai_call,
)
from market_context import (
    PricePressure,
    OrderBookPressure,
    ChartStructure,
    NewsFetchResult,
    ResistanceZoneAnalysis,
    RetestAssessment,
    ScenarioLevels,
    TimeframeConsensus,
    assess_breakout_retest,
    analyze_resistance_zone,
    analyze_price_pressure,
    build_consensus,
    build_scenario_levels,
    fetch_market_news,
    fetch_order_book_pressure,
    find_chart_structure,
    news_risk_label,
    render_multi_timeframe_chart,
    select_closed_candles,
)
from realtime_price import BinanceFuturesPriceStream, RealtimeQuote
from peak_analysis import (
    LiquidityTrapAssessment,
    PeakExecutionPlan,
    PeakLiquidityAssessment,
    PeakMap,
    PeakTradeGate,
    SetupQualityAssessment,
    assess_liquidity_traps,
    assess_peak_liquidity,
    assess_peak_trade_gate,
    assess_setup_quality,
    build_peak_execution_plan,
    build_peak_map,
    format_peak_map,
    render_peak_confirmation_chart,
)
from macro_risk import MacroRiskAssessment, assess_macro_risk

load_dotenv()

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logger = logging.getLogger("gold-query-bot")

SIGNAL_LABEL = {"BUY": "nghieng TANG", "SELL": "nghieng GIAM", "HOLD": "trung lap, chua ro huong"}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_current_quote(provider, price_stream) -> RealtimeQuote:
    stream_quote = price_stream.latest()
    if stream_quote is not None:
        stream_age = (
            datetime.now(timezone.utc) - stream_quote.received_at
        ).total_seconds()
        if stream_age > 15:
            logger.warning(
                "Ignoring stale Binance WebSocket quote age=%.1fs; using REST",
                stream_age,
            )
            stream_quote = None
    quote = provider.get_quote()
    quote_timestamp = int(quote.get("last_quote_at") or quote.get("timestamp") or 0)
    market_open_value = quote.get("is_market_open", False)
    is_market_open = (
        market_open_value
        if isinstance(market_open_value, bool)
        else str(market_open_value).strip().lower() == "true"
    )
    rest_market_time = datetime.fromtimestamp(quote_timestamp, tz=timezone.utc)
    return RealtimeQuote(
        price=(stream_quote.price if stream_quote is not None else float(quote["close"])),
        market_time=(
            stream_quote.market_time if stream_quote is not None else rest_market_time
        ),
        received_at=datetime.now(timezone.utc),
        source=(
            "Binance Futures WebSocket + REST metrics"
            if stream_quote is not None
            else str(quote.get("source", "REST fallback"))
        ),
        is_market_open=is_market_open,
        mark_price=(
            float(quote["mark_price"])
            if quote.get("mark_price") is not None
            else None
        ),
        index_price=(
            float(quote["index_price"])
            if quote.get("index_price") is not None
            else None
        ),
        bid_price=(
            stream_quote.bid_price
            if stream_quote is not None and stream_quote.bid_price is not None
            else float(quote["bid"])
            if quote.get("bid") is not None
            else None
        ),
        ask_price=(
            stream_quote.ask_price
            if stream_quote is not None and stream_quote.ask_price is not None
            else float(quote["ask"])
            if quote.get("ask") is not None
            else None
        ),
        funding_rate=(
            float(quote["funding_rate"])
            if quote.get("funding_rate") is not None
            else None
        ),
        next_funding_time=(
            datetime.fromtimestamp(
                int(quote["next_funding_time"]),
                tz=timezone.utc,
            )
            if quote.get("next_funding_time") is not None
            else None
        ),
        open_interest=(
            float(quote["open_interest"])
            if quote.get("open_interest") is not None
            else None
        ),
    )


def get_fast_quote(provider, price_stream) -> RealtimeQuote:
    """Prefer the public WebSocket and use one lightweight REST call only if stale."""
    stream_quote = price_stream.latest(max_received_age_seconds=15)
    if stream_quote is not None:
        return stream_quote
    now = datetime.now(timezone.utc)
    price = float(provider.get_latest_price())
    return RealtimeQuote(
        price=price,
        market_time=now,
        received_at=now,
        source="Binance Futures REST ticker fallback",
        is_market_open=True,
    )


COMPONENT_LABEL = {"rsi": "RSI momentum", "macd": "MACD", "ema_trend": "EMA trend", "bollinger": "Bollinger %B"}


@dataclass
class TradingPlan:
    side: str
    entry: float
    stop: float
    take_profit_1: float
    take_profit_2: float
    risk_usdt: float
    quantity_xau: float
    notional_usdt: float
    leverage: int
    margin_usdt: float
    margin_pct: float
    actual_risk_pct: float
    account_balance_usdt: float
    holding_style: str
    max_holding_minutes: int
    allow_overnight: bool


@dataclass
class PeakOpportunity:
    analysis_now: datetime
    frames: dict
    realtime_quote: RealtimeQuote
    peak_map: PeakMap
    liquidity: PeakLiquidityAssessment
    trap: LiquidityTrapAssessment
    macro_risk: MacroRiskAssessment
    hourly_structure: ChartStructure
    daily_structure: ChartStructure
    momentum_biases: dict[str, MomentumBias]
    momentum_scores: dict[str, float]
    gate: PeakTradeGate
    execution_plan: PeakExecutionPlan | None
    execution_reason: str
    quality: SetupQualityAssessment
    sized_plan: TradingPlan | None


def interval_to_minutes(interval: str) -> int:
    units = {"min": 1, "h": 60, "day": 1440}
    for suffix, multiplier in units.items():
        if interval.endswith(suffix):
            return max(1, int(float(interval[: -len(suffix)]) * multiplier))
    return 15


def format_duration(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes} phut"
    hours = minutes / 60
    return f"{hours:.0f} gio" if hours.is_integer() else f"{hours:.1f} gio"


def build_trading_plan(
    bias: MomentumBias,
    result: SignalResult,
    settings: dict,
    interval: str = "15min",
    actionable: bool = True,
    side_override: str | None = None,
    entry_override: float | None = None,
    stop_override: float | None = None,
    take_profit_1_override: float | None = None,
    take_profit_2_override: float | None = None,
) -> TradingPlan | None:
    """Size a hypothetical isolated-margin position from a fixed account-risk budget."""
    minimum_bias = float(settings.get("minimum_bias", 0.25))
    if not actionable or abs(bias.composite) < minimum_bias:
        return None

    balance = float(os.getenv("ACCOUNT_BALANCE_USDT", settings.get("account_balance_usdt", 1000)))
    risk_pct = float(settings.get("risk_per_trade_pct", 0.5))
    max_leverage = max(1, int(settings.get("max_leverage", 5)))
    max_margin_pct = float(settings.get("max_margin_pct", 25))
    if balance <= 0 or risk_pct <= 0 or max_margin_pct <= 0:
        raise ValueError("Trading-plan balance/risk/margin settings must be positive")

    side = side_override or ("LONG" if bias.composite > 0 else "SHORT")
    entry = float(entry_override if entry_override is not None else result.price)
    price_tick = max(float(settings.get("price_tick", 0.01)), 1e-9)
    entry = round(entry / price_tick) * price_tick
    stop_distance = (
        abs(entry - float(stop_override))
        if stop_override is not None
        else result.suggested_stop_distance
    )
    stop_distance = max(stop_distance, entry * 0.0001)
    risk_budget = balance * risk_pct / 100
    if "estimated_round_trip_fee_pct" in settings:
        fee_per_unit = entry * max(
            0.0,
            float(settings.get("estimated_round_trip_fee_pct", 0.10)),
        ) / 100
        slippage_per_unit = entry * max(
            0.0,
            float(settings.get("estimated_slippage_bps", 2.0)),
        ) / 10_000
        estimated_risk_per_unit = stop_distance + fee_per_unit + slippage_per_unit
    else:
        # Backward-compatible fallback for older config files.
        buffer = 1 + float(settings.get("slippage_fee_buffer_pct", 10)) / 100
        estimated_risk_per_unit = stop_distance * buffer
    quantity = risk_budget / estimated_risk_per_unit

    # Keep isolated margin below the configured account percentage, even at max leverage.
    max_margin = balance * max_margin_pct / 100
    quantity = min(quantity, max_margin * max_leverage / entry)
    quantity_step = max(float(settings.get("quantity_step", 0.001)), 1e-9)
    quantity = math.floor(quantity / quantity_step) * quantity_step
    minimum_quantity = float(settings.get("minimum_quantity", 0.001))
    minimum_notional = float(settings.get("minimum_notional_usdt", 5))
    if quantity < minimum_quantity or quantity * entry < minimum_notional:
        return None
    notional = quantity * entry
    leverage = max(1, min(max_leverage, math.ceil(notional / max_margin)))
    margin = notional / leverage
    actual_risk = quantity * estimated_risk_per_unit
    strong_bias = abs(bias.composite) >= 0.5
    holding_bars = int(
        settings.get("strong_bias_holding_bars", 16)
        if strong_bias
        else settings.get("moderate_bias_holding_bars", 8)
    )
    max_holding_minutes = interval_to_minutes(interval) * max(1, holding_bars)

    direction = 1 if side == "LONG" else -1
    stop = round((entry - direction * stop_distance) / price_tick) * price_tick
    take_profit_1 = (
        round(float(take_profit_1_override) / price_tick) * price_tick
        if take_profit_1_override is not None
        else round(
            (
                entry
                + direction
                * stop_distance
                * float(settings.get("take_profit_1_r", 1.0))
            )
            / price_tick
        )
        * price_tick
    )
    take_profit_2 = (
        round(float(take_profit_2_override) / price_tick) * price_tick
        if take_profit_2_override is not None
        else round(
            (
                entry
                + direction
                * stop_distance
                * float(settings.get("take_profit_2_r", 2.0))
            )
            / price_tick
        )
        * price_tick
    )
    return TradingPlan(
        side=side,
        entry=entry,
        stop=stop,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        risk_usdt=actual_risk,
        quantity_xau=quantity,
        notional_usdt=notional,
        leverage=leverage,
        margin_usdt=margin,
        margin_pct=margin / balance * 100,
        actual_risk_pct=actual_risk / balance * 100,
        account_balance_usdt=balance,
        holding_style="ngan han trong ngay",
        max_holding_minutes=max_holding_minutes,
        allow_overnight=bool(settings.get("allow_overnight", False)),
    )


def format_peak_execution_guide(
    gate: PeakTradeGate,
    execution_plan: PeakExecutionPlan | None,
    execution_reason: str,
    sized_plan: TradingPlan | None,
    liquidity: PeakLiquidityAssessment,
    hourly_structure: ChartStructure,
    hourly_score: float,
    daily_structure: ChartStructure,
    daily_score: float,
    quality: SetupQualityAssessment,
    trap: LiquidityTrapAssessment,
    macro_risk: MacroRiskAssessment,
) -> str:
    resistance = gate.resistance
    ratio_text = (
        f"{liquidity.volume_ratio:.0%} median ngày thường"
        if liquidity.volume_ratio is not None
        else "chưa đủ mẫu so sánh"
    )
    hourly_status = (
        "CHƯA XÉT"
        if not gate.multi_timeframe_aligned
        else "ĐẠT"
        if gate.hourly_confirmed
        else "CHƯA ĐẠT"
    )
    daily_status = (
        "CHƯA XÉT"
        if not gate.multi_timeframe_aligned
        else "ĐẠT"
        if gate.daily_confirmed
        else "CHƯA ĐẠT"
    )
    lines = [
        "🎯 HƯỚNG DẪN THỰC THI",
        f"• Chất lượng setup: {quality.score}/100 · {quality.tier} · "
        f"{quality.recommendation} (đây không phải xác suất thắng).",
        *(
            ["• PAPER ONLY: strategy live chưa đủ mẫu để xác nhận edge sau phí."]
            if quality.paper_only
            else []
        ),
        f"• Sweep/FOMO: {trap.reason}.",
        f"• Tin vĩ mô: {macro_risk.level} · {macro_risk.reason}",
        f"• 1H: {hourly_structure.pattern} · bias {hourly_score * 100:+.0f}% · "
        f"S {hourly_structure.support:.2f} / R {hourly_structure.resistance:.2f} · "
        f"xác nhận {hourly_status}.",
        f"• D1: {daily_structure.pattern} · bias {daily_score * 100:+.0f}% · "
        f"S {daily_structure.support:.2f} / R {daily_structure.resistance:.2f} · "
        f"không đối nghịch {daily_status}.",
        f"• Thanh khoản: {liquidity.status} · volume {ratio_text}.",
    ]
    if resistance is None:
        return "\n".join(lines + ["• KHÔNG VÀO: chưa có vùng cản đủ tin cậy để lập kế hoạch."])

    if execution_plan is None:
        if quality.blockers:
            lines.append("• Blocker scorecard: " + " | ".join(quality.blockers[:3]))
        lines += [
            f"• Hiện tại: KHÔNG VÀO — {execution_reason}",
            f"• SHORT: chờ giá hồi vào {resistance.lower:.2f}–{resistance.upper:.2f}; "
            f"15m từ chối, 1H đóng dưới {resistance.lower:.2f}, D1 không tăng mạnh; rồi retest không vượt lại vùng.",
            f"• LONG: chờ 15m đóng trên {resistance.upper:.2f}; nến sau retest "
            f"{resistance.lower:.2f}–{resistance.upper:.2f}, 1H đóng trên cản và D1 không giảm mạnh.",
            "• Lúc này chưa đặt Limit/SL/TP. Đủ điều kiện thì gọi /dinh lại; râu nến không tính xác nhận.",
            "• Khi bot có kế hoạch: Entry bằng Limit → khớp xong mới đặt Stop-Market toàn vị thế, trigger Mark Price.",
            "• Luật SL: tối đa khoảng 7 giá từ Entry thực tế; nếu cấu trúc cần xa hơn thì BỎ KÈO, không ép SL sai vị trí.",
            "• Retest thất bại sau khi khớp: cắt sớm ngay khi bot báo; không cố chờ giá chạy đủ 7 giá mới thoát.",
            "• TP1 đóng 50%, TP2 đóng phần còn lại; cả hai phải là Close/Reduce-Only để không mở lệnh ngược.",
        ]
        return "\n".join(lines)

    plan = execution_plan
    retest_action = (
        "giữ trên vùng phá"
        if plan.side == "LONG"
        else "không vượt lại cản"
    )
    entry_button = "Mua/Long" if plan.side == "LONG" else "Bán/Short"
    close_button = "Bán/Close Long" if plan.side == "LONG" else "Mua/Close Short"
    stop_distance = abs(plan.entry_reference - plan.stop_loss)
    lines += [
        f"• {plan.side}: chờ giá quay lại {plan.entry_lower:.2f}–{plan.entry_upper:.2f} "
        f"và nến 15m retest {retest_action}; 1H xác nhận, D1 không đối nghịch, không đuổi giá.",
        f"• CẮT LỖ CỨNG: SL {plan.stop_loss:.2f}, cách Entry tham chiếu "
        f"{stop_distance:.2f} giá (không quá khoảng 7 giá).",
        "• Không chờ bot để cắt SL: Entry khớp là đặt Stop-Market ngay trên Binance, ưu tiên trigger Mark Price.",
        "• RETEST THẤT BẠI: nếu nến 1m đóng xuyên mép sai của vùng Entry kèm áp lực ngược, bot báo CẮT NGAY; đóng toàn bộ Reduce-Only, không chờ hard SL.",
        f"• TP1 {plan.take_profit_1:.2f} ({plan.reward_risk_1:.1f}R) · "
        f"TP2 {plan.take_profit_2:.2f} ({plan.reward_risk_2:.1f}R, {plan.structural_target}).",
    ]
    if sized_plan is not None:
        lines += [
            f"• Khối lượng mô phỏng: {sized_plan.quantity_xau:.3f} XAU · {sized_plan.leverage}x isolated · "
            f"rủi ro mô phỏng tối đa {sized_plan.risk_usdt:.2f} USDT "
            f"({sized_plan.actual_risk_pct:.2f}% trên số dư cấu hình "
            f"{sized_plan.account_balance_usdt:.2f} USDT).",
            f"• Trần risk theo mức {quality.tier}: {quality.recommended_risk_pct:.2f}% tài khoản; "
            "không tăng size vì thấy điểm 90+.",
            f"• Quản lệnh: chốt 50% ở TP1, phần còn lại dời SL về Entry; "
            f"time-stop ~{format_duration(sized_plan.max_holding_minutes)}, không giữ qua ngày.",
        ]
        if quality.paper_only:
            lines += [
                "",
                "CÁCH GHI PAPER TRADE",
                f"1) Ghi Limit giả lập {entry_button} {sized_plan.quantity_xau:.3f} XAU trong vùng Entry; không đặt lệnh thật.",
                f"2) Chỉ đánh dấu fill khi nến sau tín hiệu chạm giá {plan.entry_reference:.2f}; ghi SL {plan.stop_loss:.2f} ngay trong journal.",
                f"3) Ghi TP1 50% tại {plan.take_profit_1:.2f}, TP2 phần còn lại tại {plan.take_profit_2:.2f}; tính đủ phí, spread và trượt giá.",
                "4) Lưu ảnh/nến, thời gian, điểm setup, lý do thoát; không chuyển paper signal thành lệnh thật.",
            ]
        else:
            lines += [
                "",
                "CÁCH ĐẶT TRÊN BINANCE",
                f"1) Chọn Isolated {sized_plan.leverage}x → Limit {entry_button} "
                f"{sized_plan.quantity_xau:.3f} XAU trong vùng Entry; hết vùng thì hủy, không Market đuổi.",
                f"2) NGAY KHI Entry khớp → đặt Stop-Market {close_button} toàn bộ vị thế tại "
                f"{plan.stop_loss:.2f}; trigger Mark Price và chọn Close Position/Reduce-Only. Không đặt SL sau.",
                f"3) Đặt TP Limit {close_button}: 50% tại {plan.take_profit_1:.2f}; "
                f"phần còn lại tại {plan.take_profit_2:.2f}, đều chỉ giảm vị thế.",
                "4) Nếu bot báo RETEST THẤT BẠI → Market Close/Close Position toàn bộ ngay, Reduce-Only; hủy TP/SL còn treo và không đảo chiều.",
                "5) TP1 khớp → dời SL phần còn lại về Entry; vị thế đã đóng thì hủy mọi lệnh SL/TP còn treo.",
            ]
    else:
        lines.append("• Không tính được khối lượng an toàn với số dư/cấu hình hiện tại; không vào lệnh.")
    lines.append("• Hủy kế hoạch nếu nến 15m đóng phá SL trước khi khớp Entry.")
    return "\n".join(lines)


def compute_peak_opportunity(
    config: dict,
    provider,
    price_stream,
    analysis_now: datetime | None = None,
) -> PeakOpportunity:
    """Compute the exact /dinh opportunity for both manual queries and auto alerts."""
    settings = config.get("peak_map", {})
    timeframes = settings.get("timeframes", ["15min", "1h", "4h", "1day"])
    outputsize = int(settings.get("outputsize", 200))
    outputsize_by_timeframe = settings.get("outputsize_by_timeframe", {})
    analysis_now = analysis_now or datetime.now(timezone.utc)
    frames = {
        timeframe: select_closed_candles(
            provider.get_historical(
                interval=timeframe,
                outputsize=max(
                    40,
                    min(
                        1500,
                        int(outputsize_by_timeframe.get(timeframe, outputsize)),
                    ),
                ),
            ),
            timeframe,
            analysis_now,
        )
        for timeframe in timeframes
    }
    realtime_quote = get_current_quote(provider, price_stream)
    peak_map_frames = {
        timeframe: frame.tail(outputsize)
        for timeframe, frame in frames.items()
    }
    peak_map = build_peak_map(
        frames=peak_map_frames,
        current_price=realtime_quote.price,
        settings=settings,
    )
    liquidity = assess_peak_liquidity(
        frames["1h"],
        realtime_quote.price,
        realtime_quote.bid_price,
        realtime_quote.ask_price,
        analysis_now=analysis_now,
        settings=config.get("peak_liquidity", {}),
    )
    trap = assess_liquidity_traps(
        peak_map,
        frames["15min"],
        config.get("liquidity_traps", {}),
    )
    macro_risk = assess_macro_risk(
        analysis_now,
        config.get("macro_guard", {}),
    )
    hourly_structure = find_chart_structure(frames["1h"])
    daily_structure = find_chart_structure(frames["1day"])
    analysis_frames = {
        timeframe: frame
        for timeframe, frame in frames.items()
        if timeframe in ("15min", "1h", "4h", "1day")
    }
    momentum_biases = {
        timeframe: compute_momentum_bias(frame, config["weights"])
        for timeframe, frame in analysis_frames.items()
    }
    momentum_scores = {
        timeframe: bias.composite
        for timeframe, bias in momentum_biases.items()
    }
    gate = assess_peak_trade_gate(
        peak_map,
        frames["15min"],
        momentum_scores,
        {
            **settings,
            **config.get("signal_confirmation", {}),
            **config.get("peak_execution", {}),
        },
        frame_1h=frames["1h"],
        liquidity=liquidity,
        daily_pattern=daily_structure.pattern,
        trap=trap,
        macro_risk=macro_risk,
    )
    execution_plan, execution_reason = build_peak_execution_plan(
        peak_map,
        gate,
        frames["15min"],
        {
            **config.get("trading_plan", {}),
            **config.get("peak_execution", {}),
        },
    )
    quality_settings = {
        **config.get("setup_quality", {}),
        "base_risk_per_trade_pct": config.get("trading_plan", {}).get(
            "risk_per_trade_pct",
            0.5,
        ),
    }
    quality = assess_setup_quality(
        gate,
        execution_plan,
        liquidity,
        trap,
        macro_risk,
        quality_settings,
    )
    sized_plan = None
    if execution_plan is not None and quality.actionable:
        signal_result = compute_signal(
            frames["15min"],
            config["weights"],
            {
                "buy": config["threshold_buy"],
                "sell": config["threshold_sell"],
            },
            atr_stop_multiplier=config.get("atr_stop_multiplier", 1.5),
        )
        sized_plan = build_trading_plan(
            momentum_biases["15min"],
            signal_result,
            {
                **config.get("trading_plan", {}),
                "minimum_bias": 0.0,
                "risk_per_trade_pct": quality.recommended_risk_pct,
            },
            actionable=True,
            side_override=execution_plan.side,
            # Size from the worst fill edge of the zone, not its optimistic midpoint.
            entry_override=(
                execution_plan.entry_upper
                if execution_plan.side == "LONG"
                else execution_plan.entry_lower
            ),
            stop_override=execution_plan.stop_loss,
            take_profit_1_override=execution_plan.take_profit_1,
            take_profit_2_override=execution_plan.take_profit_2,
        )
    return PeakOpportunity(
        analysis_now=analysis_now,
        frames=frames,
        realtime_quote=realtime_quote,
        peak_map=peak_map,
        liquidity=liquidity,
        trap=trap,
        macro_risk=macro_risk,
        hourly_structure=hourly_structure,
        daily_structure=daily_structure,
        momentum_biases=momentum_biases,
        momentum_scores=momentum_scores,
        gate=gate,
        execution_plan=execution_plan,
        execution_reason=execution_reason,
        quality=quality,
        sized_plan=sized_plan,
    )


def format_reply(bias: MomentumBias, result: SignalResult, interval: str, plan_settings: dict) -> str:
    # ATR duoc tinh tren khung nen live (mac dinh 15min). Dung quy tac can bac hai thoi gian
    # (random-walk scaling) de uoc luong bien dong ky vong cho 1h va 4h toi - chi la uoc luong
    # thong ke, KHONG phai cam ket huong di.
    bars_per_hour = {"1min": 60, "5min": 12, "15min": 4, "30min": 2, "45min": 1.33, "1h": 1}
    per_hour = bars_per_hour.get(interval, 4)
    move_1h = result.atr * (per_hour ** 0.5)
    move_4h = result.atr * ((per_hour * 4) ** 0.5)
    pct = bias.composite * 100

    lines = [
        f"*Binance XAUUSDT PERP*: {bias.price:.2f} USDT/oz",
        f"Xu huong ngan han: *{bias.label}* (bias {pct:+.0f}%)",
        "",
        "Chi tiet (moi chi bao tu -100% den +100%):",
    ]
    for key, val in bias.components.items():
        lines.append(f"  - {COMPONENT_LABEL[key]}: {val * 100:+.0f}%")
    plan = build_trading_plan(bias, result, plan_settings, interval)
    lines += [
        "",
        f"Bien dong uoc luong: ~1h toi +/-{move_1h:.2f} USD, ~4h toi +/-{move_4h:.2f} USD quanh gia hien tai.",
        "",
        f"(Tham khao) Tin hieu backtest cu, it kich hoat hon: *{result.signal}* ({SIGNAL_LABEL[result.signal]})",
    ]
    if plan is None:
        lines += [
            "",
            "*KE HOACH: KHONG VAO LENH*",
            f"Bias chua dat {float(plan_settings.get('minimum_bias', 0.25)) * 100:.0f}%; dung ngoai de tranh ep lenh khi thi truong khong ro huong.",
        ]
    else:
        lines += [
            "",
            "*KE HOACH QUAN TRI VON (tham khao)*",
            f"Hướng: *{plan.side}* | Entry tham khao: {plan.entry:.2f}",
            f"Stop-loss: *{plan.stop:.2f}* | TP1: {plan.take_profit_1:.2f} (1R) | TP2: *{plan.take_profit_2:.2f}* (2R)",
            f"Rui ro toi da du kien: *{plan.risk_usdt:.2f} USDT* ({plan.actual_risk_pct:.2f}% tai khoan, da dem buffer phi/truot gia)",
            f"Khoi luong: *{plan.quantity_xau:.4f} XAU* | Gia tri vi the: {plan.notional_usdt:.2f} USDT",
            f"Don bay de xuat: *{plan.leverage}x isolated* | Ky quy: {plan.margin_usdt:.2f} USDT ({plan.margin_pct:.1f}% tai khoan)",
            f"Thoi gian: *{plan.holding_style}*, toi da ~{format_duration(plan.max_holding_minutes)}; "
            + ("co the giu qua ngay neu danh gia lai." if plan.allow_overnight else "*khong giu qua ngay*."),
            "Chot 50% tai TP1; neu da khop TP1, co the doi stop phan con lai ve entry. Bo lenh neu gia da chay qua entry.",
            "Time-stop: het thoi gian tren ma chua cham TP/SL thi dong lenh hoac tinh lai tin hieu; khong tu bien lenh ngan han thanh lenh dai han.",
        ]
    lines += [
        "",
        "_Day la kich ban quan tri rui ro, KHONG phai dam bao gia se di dung huong. Bias chua duoc backtest rieng; "
        "cau hinh cu chi co win rate ~47%. Khoi luong XAU va gia thanh ly phu thuoc quy cach hop dong cua san; "
        "phai doi chieu tren man hinh dat lenh truoc khi giao dich. Stop co the truot gia khi bien dong manh._",
    ]
    return "\n".join(lines)


def _safe_text(value: str) -> str:
    return value.replace("*", "").replace("_", " ").replace("`", "")


def format_enhanced_reply(
    bias: MomentumBias,
    result: SignalResult,
    interval: str,
    plan_settings: dict,
    consensus: TimeframeConsensus,
    pressure: PricePressure,
    order_book: OrderBookPressure | None,
    news_result: NewsFetchResult,
    structures: dict[str, ChartStructure],
    quote_time: datetime,
    quote_age_seconds: float,
    quote_received_age_seconds: float,
    is_market_open: bool,
    quote_source: str,
) -> tuple[str, TradingPlan | None]:
    bars_per_hour = {"1min": 60, "5min": 12, "15min": 4, "30min": 2, "45min": 1.33, "1h": 1}
    per_hour = bars_per_hour.get(interval, 4)
    move_1h = result.atr * (per_hour ** 0.5)
    move_4h = result.atr * ((per_hour * 4) ** 0.5)

    planning_bias = MomentumBias(
        price=bias.price,
        composite=consensus.score,
        label=consensus.label,
        components=bias.components,
    )
    plan = build_trading_plan(
        planning_bias,
        result,
        plan_settings,
        interval,
        actionable=consensus.actionable,
    )

    lines = [
        f"*Binance XAUUSDT PERP real-time*: {bias.price:.2f} USDT/oz",
        f"Gia nhan ~{quote_received_age_seconds:.1f}s truoc qua {quote_source}; "
        f"moc thoi gian nguon {quote_time.astimezone().strftime('%d/%m %H:%M:%S')} "
        f"(nguon lam tron/tre ~{quote_age_seconds:.0f}s) | "
        + ("thi truong dang mo" if is_market_open else "thi truong dang dong"),
        f"*KET LUAN DA KHUNG*: {consensus.label} ({consensus.score * 100:+.0f}%)",
        "",
        "Da khung thoi gian:",
    ]
    for timeframe in ("15min", "1h", "4h"):
        lines.append(f"  - {timeframe}: {consensus.scores.get(timeframe, 0) * 100:+.0f}%")
    lines += ["", "Cau truc dinh/day:"]
    for timeframe in ("15min", "1h", "4h"):
        structure = structures[timeframe]
        lines.append(
            f"  - {timeframe}: {structure.pattern}, {structure.trend}; "
            f"ho tro {structure.support:.2f}, khang cu {structure.resistance:.2f}"
        )

    lines += [
        "",
        f"Ap luc gia 1 phut: *{pressure.label}* ({pressure.score * 100:+.0f}%)",
        f"  - {pressure.up_bars} nen tang / {pressure.down_bars} nen giam trong ~{pressure.window_minutes} phut",
        "  - Day la proxy tu bien dong gia, KHONG phai volume/order book XAU that.",
    ]
    if order_book is not None:
        lines += [
            f"Order book proxy {order_book.symbol}: *{order_book.label}* ({order_book.score * 100:+.0f}%)",
            f"  - Bid ~{order_book.bid_notional:,.0f} USDT / Ask ~{order_book.ask_notional:,.0f} USDT trong top depth.",
            "  - Day la vang token hoa tren Binance, co the lech XAU spot va co the bi spoof.",
        ]
    else:
        lines.append("Order book proxy: khong lay duoc du lieu.")
    lines += ["", "Chi tiet chi bao khung 15m:"]
    for key, value in bias.components.items():
        lines.append(f"  - {COMPONENT_LABEL[key]}: {value * 100:+.0f}%")

    lines += [
        "",
        f"Bien dong uoc luong: ~1h +/-{move_1h:.2f} USD, ~4h +/-{move_4h:.2f} USD.",
        f"Tin hieu backtest cu: *{result.signal}* ({SIGNAL_LABEL[result.signal]})",
    ]
    if plan is None:
        lines += [
            "",
            "*KE HOACH: KHONG VAO LENH*",
            "Khung 15m, 1H va 4H chua dong thuan hoac diem tong hop chua du manh. Cho tin hieu moi, khong ep lenh.",
        ]
    else:
        lines += [
            "",
            "*KE HOACH QUAN TRI VON (tham khao)*",
            f"Huong: *{plan.side}* | Entry: {plan.entry:.2f}",
            f"Stop-loss: *{plan.stop:.2f}* | TP1: {plan.take_profit_1:.2f} (1R) | TP2: *{plan.take_profit_2:.2f}* (2R)",
            f"Rui ro du kien: *{plan.risk_usdt:.2f} USDT* ({plan.actual_risk_pct:.2f}% tai khoan, co buffer phi/truot gia)",
            f"Khoi luong: *{plan.quantity_xau:.4f} XAU* | Gia tri vi the: {plan.notional_usdt:.2f} USDT",
            f"Don bay: *{plan.leverage}x isolated* | Ky quy: {plan.margin_usdt:.2f} USDT ({plan.margin_pct:.1f}% tai khoan)",
            f"Thoi gian: *{plan.holding_style}*, toi da ~{format_duration(plan.max_holding_minutes)}; "
            + ("co the giu qua ngay neu danh gia lai." if plan.allow_overnight else "*khong giu qua ngay*."),
            "Chot 50% tai TP1, sau do co the doi stop phan con lai ve entry.",
            "Time-stop: het thoi gian ma chua cham TP/SL thi dong lenh hoac tinh lai; khong gong thanh lenh dai han.",
        ]

    news_items = news_result.items
    news_status = f"vua cap nhat truc tiep {news_result.fetched_at.astimezone().strftime('%H:%M:%S')}"
    lines += ["", f"Rui ro tin tuc: *{news_risk_label(news_items)}* ({news_status})"]
    if news_items:
        for item in news_items[:4]:
            lines.append(f"  - [{item.source}] {_safe_text(item.title)}")
    else:
        source_count = len(news_result.source_errors)
        lines.append(f"  - Chua lay duoc tin truc tiep ({source_count} nguon loi). Khong xem day la xac nhan an toan.")

    lines += [
        "",
        "_Day la kich ban quan tri rui ro, khong dam bao huong gia. Tin tuc chi dung de canh bao bien dong, "
        "khong tu dong suy dien LONG/SHORT. Khoi luong/price liquidation phai doi chieu theo dung hop dong cua san._",
    ]
    return "\n".join(lines), plan


def format_compact_reply(
    bias: MomentumBias,
    result: SignalResult,
    interval: str,
    plan_settings: dict,
    consensus: TimeframeConsensus,
    pressure: PricePressure,
    order_book: OrderBookPressure | None,
    news_result: NewsFetchResult,
    structures: dict[str, ChartStructure],
    resistance_test: ResistanceZoneAnalysis,
    retest: RetestAssessment,
    realtime_quote: RealtimeQuote,
    liquidity: PeakLiquidityAssessment | None = None,
    macro_risk: MacroRiskAssessment | None = None,
) -> tuple[str, TradingPlan | None, ScenarioLevels]:
    planning_bias = MomentumBias(
        price=bias.price,
        composite=consensus.score,
        label=consensus.label,
        components=bias.components,
    )
    plan = None
    if (
        retest.actionable_side
        and retest.entry_lower is not None
        and retest.entry_upper is not None
        and retest.invalidation is not None
        and (liquidity is None or liquidity.entries_allowed)
        and (macro_risk is None or not macro_risk.blocked)
    ):
        plan = build_trading_plan(
            planning_bias,
            result,
            plan_settings,
            interval,
            actionable=True,
            side_override=retest.actionable_side,
            entry_override=(retest.entry_lower + retest.entry_upper) / 2,
            stop_override=retest.invalidation,
        )
    scenarios = build_scenario_levels(bias.price, result.atr, structures)

    forecast = plan.side if plan is not None else "CHỜ"

    pressure_side = "mua" if pressure.score > 0.08 else "bán" if pressure.score < -0.08 else "cân bằng"
    pressure_source = "taker flow thật" if pressure.uses_trade_flow else "price action"
    order_book_text = "không có"
    if order_book is not None:
        order_side = "mua" if order_book.score > 0.08 else "bán" if order_book.score < -0.08 else "cân bằng"
        order_book_text = f"{order_side} {order_book.score * 100:+.0f}%"

    news_items = news_result.items
    news_risk = "CAO" if news_risk_label(news_items).startswith("CAO") else "BÌNH THƯỜNG"
    news_status = f"trực tiếp {news_result.fetched_at.astimezone().strftime('%H:%M:%S')}"
    volume_text = (
        f"{resistance_test.volume_ratio:.2f}x trung bình 20 nến"
        if resistance_test.volume_ratio is not None
        else "chưa có volume hợp lệ"
    )
    quote_time = realtime_quote.market_time
    quote_market_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - quote_time).total_seconds(),
    )
    quote_received_age_seconds = max(
        0.0,
        (datetime.now(timezone.utc) - realtime_quote.received_at).total_seconds(),
    )
    if quote_market_age_seconds < 90:
        market_age = f"{quote_market_age_seconds:.0f}s"
    elif quote_market_age_seconds < 3600:
        market_age = f"{quote_market_age_seconds / 60:.0f} phút"
    elif quote_market_age_seconds < 86400:
        market_age = f"{quote_market_age_seconds / 3600:.1f} giờ"
    else:
        market_age = f"{quote_market_age_seconds / 86400:.1f} ngày"
    if not realtime_quote.is_market_open:
        feed_status = f"ĐÓNG CỬA · giá cuối cách {market_age}"
    elif quote_market_age_seconds > 120 or quote_received_age_seconds > 120:
        feed_status = f"DỮ LIỆU TRỄ {market_age}"
    else:
        feed_status = f"LIVE {quote_received_age_seconds:.1f}s"

    fair_value_lines = []
    if realtime_quote.mark_price is not None and realtime_quote.index_price is not None:
        basis = realtime_quote.mark_price - realtime_quote.index_price
        fair_value_lines.append(
            f"Mark {realtime_quote.mark_price:.2f} · Index {realtime_quote.index_price:.2f} · basis {basis:+.2f}"
        )
    if realtime_quote.bid_price is not None and realtime_quote.ask_price is not None:
        spread = realtime_quote.ask_price - realtime_quote.bid_price
        fair_value_lines.append(
            f"Bid {realtime_quote.bid_price:.2f} · Ask {realtime_quote.ask_price:.2f} · spread {spread:.2f}"
        )
    derivatives_parts = []
    if realtime_quote.funding_rate is not None:
        derivatives_parts.append(f"funding {realtime_quote.funding_rate * 100:+.4f}%")
    if realtime_quote.next_funding_time is not None:
        derivatives_parts.append(
            "kỳ tới " + realtime_quote.next_funding_time.astimezone().strftime("%H:%M")
        )
    if realtime_quote.open_interest is not None:
        derivatives_parts.append(f"OI {realtime_quote.open_interest:,.0f}")
    if derivatives_parts:
        fair_value_lines.append(" · ".join(derivatives_parts))

    decision_reason = (
        macro_risk.reason
        if macro_risk is not None and macro_risk.blocked
        else liquidity.reason
        if liquidity is not None and not liquidity.entries_allowed
        else retest.decision_reason
    )
    liquidity_line = (
        f"Thanh khoản: {liquidity.status} · volume 1H "
        + (
            f"{liquidity.volume_ratio:.0%} median ngày thường."
            if liquidity.volume_ratio is not None
            else "chưa đủ mẫu so sánh."
        )
        if liquidity is not None
        else None
    )

    lines = [
        f"📊 *XAUUSDT PERP {bias.price:.2f}* · {feed_status} · {realtime_quote.source}",
        f"Nguồn lúc {quote_time.astimezone().strftime('%d/%m %H:%M:%S')}",
        *fair_value_lines,
        *([liquidity_line] if liquidity_line is not None else []),
        *(
            [f"Tin vĩ mô: {macro_risk.level} · {macro_risk.reason}"]
            if macro_risk is not None
            else []
        ),
        f"🧭 *QUYẾT ĐỊNH: {forecast}* — {decision_reason}",
        "",
        "🧱 *VÙNG GIÁ*",
        f"• Hỗ trợ: *{retest.support.lower:.2f}–{retest.support.upper:.2f}*",
        f"• Kháng cự: *{retest.resistance.lower:.2f}–{retest.resistance.upper:.2f}*",
        "",
        "🎯 *CHỜ ĐIỂM VÀO*",
        f"• LONG: {retest.long_phase}; retest giữ R rồi mới vào.",
        f"• SHORT: {retest.short_phase}; retest không lấy lại S rồi mới vào.",
    ]
    if plan is not None:
        lines += [
            "",
            "🛡 *KẾ HOẠCH ĐÃ XÁC NHẬN*",
            f"• {plan.side}: vào {retest.entry_lower:.2f}–{retest.entry_upper:.2f} · SL {plan.stop:.2f}.",
            f"• TP1 {plan.take_profit_1:.2f} · TP2 {plan.take_profit_2:.2f} · rủi ro {plan.actual_risk_pct:.2f}% · {plan.leverage}x.",
        ]
    else:
        lines += [
            "",
            "🛡 *HIỆN CHƯA CÓ ENTRY* — không đuổi giá, không vào giữa vùng.",
        ]

    lines += [
        "",
        f"🔎 Nến đóng: 15m {consensus.scores.get('15min', 0) * 100:+.0f}% · "
        f"1H {consensus.scores.get('1h', 0) * 100:+.0f}% · "
        f"4H {consensus.scores.get('4h', 0) * 100:+.0f}% · RSI {resistance_test.rsi14:.0f}.",
        f"Áp lực 1m {pressure_side} {pressure.score * 100:+.0f}% ({pressure_source}) · "
        f"Order book XAU {order_book_text} · volume {volume_text}.",
        f"Tin tức: {news_risk} ({news_status}).",
        "_Chỉ dùng nến đã đóng để xác nhận; vùng giá không bảo đảm đảo chiều._",
    ]
    return "\n".join(lines), plan, scenarios


async def handle_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config = context.bot_data["config"]
    provider = context.bot_data["provider"]
    price_stream = context.bot_data["price_stream"]
    try:
        interval = config["live"]["interval"]
        timeframes = config["live"].get("analysis_timeframes", [interval, "1h", "4h"])
        analysis_now = datetime.now(timezone.utc)
        raw_frames = {
            timeframe: provider.get_historical(
                interval=timeframe,
                outputsize=config["live"]["outputsize"],
            )
            for timeframe in timeframes
        }
        frames = {
            timeframe: select_closed_candles(frame, timeframe, analysis_now)
            for timeframe, frame in raw_frames.items()
        }
        df = frames[interval]
        biases = {
            timeframe: compute_momentum_bias(frame, config["weights"])
            for timeframe, frame in frames.items()
        }
        structures = {
            timeframe: find_chart_structure(frame)
            for timeframe, frame in frames.items()
        }
        bias = biases[interval]
        pressure_interval = config["live"].get("pressure_interval", "1min")
        pressure_df = select_closed_candles(
            provider.get_historical(
            interval=config["live"].get("pressure_interval", "1min"),
            outputsize=config["live"].get("pressure_bars", 30),
            ),
            pressure_interval,
            analysis_now,
        )
        pressure = analyze_price_pressure(pressure_df)
        order_book_config = config.get(
            "order_book",
            config.get("order_book_proxy", {}),
        )
        order_book = None
        if order_book_config.get("enabled", True):
            order_book = fetch_order_book_pressure(
                order_book_config.get(
                    "endpoint",
                    "https://fapi.binance.com/fapi/v1/depth",
                ),
                order_book_config.get("symbol", "XAUUSDT"),
                int(order_book_config.get("depth_limit", 100)),
            )
        pressure_input = pressure.score if order_book is None else 0.6 * pressure.score + 0.4 * order_book.score
        consensus = build_consensus(
            {timeframe: value.composite for timeframe, value in biases.items()},
            pressure_input,
            settings=config.get("signal_confirmation", {}),
        )
        result = compute_signal(
            df,
            config["weights"],
            {"buy": config["threshold_buy"], "sell": config["threshold_sell"]},
            atr_stop_multiplier=config.get("atr_stop_multiplier", 1.5),
        )
        signal_series = compute_signal_series(
            df,
            config["weights"],
            {"buy": config["threshold_buy"], "sell": config["threshold_sell"]},
        )
        recent_signal_bars = int(
            config.get("signal_confirmation", {}).get("recent_backtest_bars", 8)
        )
        recent_signals = set(signal_series["signal"].tail(recent_signal_bars))
        legacy_long_confirmed = "BUY" in recent_signals
        legacy_short_confirmed = "SELL" in recent_signals
        realtime_quote = get_current_quote(provider, price_stream)
        if realtime_quote.open_interest is None and hasattr(
            provider,
            "get_open_interest",
        ):
            try:
                realtime_quote.open_interest = provider.get_open_interest()
            except Exception:
                logger.warning("Could not fetch Binance Futures open interest")
        latest_price = realtime_quote.price
        bias.price = latest_price
        result.price = latest_price
        liquidity = assess_peak_liquidity(
            frames["1h"],
            latest_price,
            realtime_quote.bid_price,
            realtime_quote.ask_price,
            analysis_now=analysis_now,
            settings=config.get("peak_liquidity", {}),
        )
        macro_risk = assess_macro_risk(
            analysis_now,
            config.get("macro_guard", {}),
        )
        resistance_test = analyze_resistance_zone(
            df=df,
            current_price=latest_price,
            atr=result.atr,
            structures=structures,
            settings=config.get("resistance_test", {}),
        )
        retest = assess_breakout_retest(
            df=df,
            current_price=latest_price,
            atr=result.atr,
            consensus=consensus,
            resistance_test=resistance_test,
            structures=structures,
            legacy_long_confirmed=legacy_long_confirmed,
            legacy_short_confirmed=legacy_short_confirmed,
            settings={
                **config.get("resistance_test", {}),
                **config.get("signal_confirmation", {}),
            },
        )
        news_config = config.get("news", {})
        news_result = fetch_market_news(
            news_config.get("feeds", []),
            limit=int(news_config.get("max_items", 4)),
        )
        reply, plan, scenarios = format_compact_reply(
            bias,
            result,
            interval,
            config.get("trading_plan", {}),
            consensus,
            pressure,
            order_book,
            news_result,
            structures,
            resistance_test,
            retest,
            realtime_quote,
            liquidity,
            macro_risk,
        )
        multi_chart = render_multi_timeframe_chart(
            frames=frames,
            structures=structures,
            resistance_test=resistance_test,
            retest_assessment=retest,
            current_price=latest_price,
            plan=plan,
        )
        charts = {"XAUUSDT 15m / 1H / 4H": multi_chart}
        logger.info(
            "Signal requested by chat %s: consensus=%.2f actionable=%s signal=%s",
            update.effective_chat.id,
            consensus.score,
            consensus.actionable,
            result.signal,
        )
        await update.message.reply_photo(
            photo=multi_chart,
            caption="Binance XAUUSDT PERP: 15m · 1H · 4H, chỉ dùng nến đã đóng.",
        )
        await update.message.reply_text(
            reply,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        ai_config = config.get("ai_analysis", {})
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        if ai_config.get("enabled", True):
            timeout_seconds = int(ai_config.get("timeout_seconds", 35))
            snapshot = build_ai_snapshot(
                df=df,
                current_price=latest_price,
                quote_source=realtime_quote.source,
                consensus=consensus,
                structures=structures,
                pressure=pressure,
                order_book=order_book,
                indicator_components=bias.components,
                atr=result.atr,
                legacy_signal=result.signal,
                news_result=news_result,
                plan=plan,
                resistance_test=resistance_test,
                frames=frames,
                retest=retest,
                derivatives_metrics={
                    "last_price": round(realtime_quote.price, 4),
                    "mark_price": (
                        round(realtime_quote.mark_price, 4)
                        if realtime_quote.mark_price is not None
                        else None
                    ),
                    "index_price": (
                        round(realtime_quote.index_price, 4)
                        if realtime_quote.index_price is not None
                        else None
                    ),
                    "bid": realtime_quote.bid_price,
                    "ask": realtime_quote.ask_price,
                    "funding_rate": realtime_quote.funding_rate,
                    "open_interest": realtime_quote.open_interest,
                    "liquidity_guard": {
                        "status": liquidity.status,
                        "is_weekend": liquidity.is_weekend,
                        "volume_ratio_vs_weekday_median": liquidity.volume_ratio,
                        "entries_allowed": liquidity.entries_allowed,
                        "reason": liquidity.reason,
                    },
                },
            )
            candidates = []
            groq_config = ai_config.get("groq", {})
            if groq_key:
                groq_model = groq_config.get("model", "qwen/qwen3.6-27b")
                candidates.append(
                    {
                        "provider": "Groq",
                        "model": groq_model,
                        "label": f"Groq · {groq_model}",
                        "api_key": groq_key,
                        "analyzer": analyze_with_groq,
                        "usage_path": groq_config.get("usage_path", "logs/groq_usage.json"),
                        "daily_budget": int(groq_config.get("daily_call_budget", 900)),
                        "cooldown_minutes": int(
                            groq_config.get("rate_limit_cooldown_minutes", 2)
                        ),
                    }
                )

            gemini_config = ai_config.get("gemini", {})
            if gemini_key:
                gemini_models = [
                    gemini_config.get("model", "gemini-3.6-flash"),
                    gemini_config.get("fallback_model", "gemini-3.5-flash-lite"),
                ]
                for gemini_model in dict.fromkeys(filter(None, gemini_models)):
                    candidates.append(
                        {
                            "provider": "Gemini",
                            "model": gemini_model,
                            "label": f"Gemini · {gemini_model}",
                            "api_key": gemini_key,
                            "analyzer": analyze_with_gemini,
                            "usage_path": gemini_config.get(
                                "usage_path", "logs/gemini_usage.json"
                            ),
                            "daily_budget": int(
                                gemini_config.get("daily_call_budget", 15)
                            ),
                            "cooldown_minutes": int(
                                gemini_config.get("rate_limit_cooldown_minutes", 60)
                            ),
                        }
                    )

            if not candidates:
                await update.message.reply_text(
                    "🤖 AI chưa được cấu hình. Thêm GROQ_API_KEY (ưu tiên) hoặc "
                    "GEMINI_API_KEY vào .env; phần định lượng phía trên vẫn là dữ liệu hiện tại."
                )
            else:
                last_error = None
                ai_result = None
                used_label = None
                unavailable_reasons = []
                for candidate in candidates:
                    blocker_key = (
                        f"ai_blocked_until:{candidate['provider']}:{candidate['model']}"
                    )
                    blocked_until = context.bot_data.get(blocker_key)
                    if (
                        blocked_until is not None
                        and datetime.now(timezone.utc) < blocked_until
                    ):
                        unavailable_reasons.append(
                            f"{candidate['label']} đang chờ hết rate limit"
                        )
                        continue
                    if not ai_daily_budget_available(
                        candidate["usage_path"], candidate["daily_budget"]
                    ):
                        unavailable_reasons.append(
                            f"{candidate['label']} đã đạt ngân sách ngày"
                        )
                        continue
                    record_ai_call(candidate["usage_path"])
                    try:
                        ai_result = await asyncio.wait_for(
                            asyncio.to_thread(
                                candidate["analyzer"],
                                candidate["api_key"],
                                candidate["model"],
                                timeout_seconds,
                                charts,
                                snapshot,
                            ),
                            timeout=timeout_seconds + 5,
                        )
                        used_label = candidate["label"]
                        break
                    except Exception as exc:
                        last_error = exc
                        if is_ai_rate_limit_error(exc):
                            context.bot_data[blocker_key] = (
                                datetime.now(timezone.utc)
                                + timedelta(minutes=candidate["cooldown_minutes"])
                            )
                            logger.warning(
                                "AI %s rate-limited; trying next provider/model",
                                candidate["label"],
                            )
                            continue
                        if candidate != candidates[-1]:
                            logger.warning(
                                "AI %s failed; trying next provider/model: %s",
                                candidate["label"],
                                type(exc).__name__,
                            )
                            continue
                        logger.exception(
                            "AI analysis failed for %s",
                            candidate["label"],
                        )
                        break

                if ai_result is not None and used_label is not None:
                    analyzed_at = datetime.now().astimezone().strftime("%H:%M:%S")
                    await update.message.reply_text(
                        format_ai_analysis(
                            ai_result,
                            used_label,
                            f"phân tích trực tiếp lúc {analyzed_at}",
                        )
                    )
                elif last_error and is_ai_rate_limit_error(last_error):
                    await update.message.reply_text(
                        "🤖 AI miễn phí đang bị giới hạn lượt nên không có phân tích AI mới. "
                        "Bot không dùng kết quả cũ; giá, biểu đồ và phần định lượng phía trên vẫn là dữ liệu hiện tại."
                    )
                elif unavailable_reasons:
                    await update.message.reply_text(
                        "🤖 AI: " + "; ".join(unavailable_reasons) + ". "
                        "Bot không dùng cache; phần định lượng phía trên vẫn là dữ liệu hiện tại."
                    )
                else:
                    await update.message.reply_text(
                        "🤖 AI: cả Groq và Gemini đều tạm lỗi nên không có phân tích AI mới. "
                        "Bot không dùng cache; phần định lượng phía trên vẫn là dữ liệu hiện tại."
                    )
    except Exception:
        logger.exception("Failed to compute signal")
        await update.message.reply_text("Co loi khi lay gia/tinh tin hieu, thu lai sau it phut.")


async def handle_peaks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config = context.bot_data["config"]
    provider = context.bot_data["provider"]
    price_stream = context.bot_data["price_stream"]
    settings = config.get("peak_map", {})
    try:
        opportunity = await asyncio.to_thread(
            compute_peak_opportunity,
            config,
            provider,
            price_stream,
        )
        frames = opportunity.frames
        realtime_quote = opportunity.realtime_quote
        peak_map = opportunity.peak_map
        liquidity = opportunity.liquidity
        hourly_structure = opportunity.hourly_structure
        daily_structure = opportunity.daily_structure
        momentum_scores = opportunity.momentum_scores
        gate = opportunity.gate
        execution_plan = opportunity.execution_plan
        execution_reason = opportunity.execution_reason
        sized_plan = opportunity.sized_plan
        reply = format_peak_map(peak_map, settings)
        logger.info(
            "Peak map requested by chat %s: peaks=%s resistance=%s support=%s",
            update.effective_chat.id,
            peak_map.scanned_peak_count,
            len(peak_map.resistance_zones),
            len(peak_map.converted_support_zones),
        )
        try:
            hourly_chart = render_peak_confirmation_chart(
                frames,
                peak_map,
                liquidity,
            )
            await update.message.reply_photo(
                photo=hourly_chart,
                caption=(
                    "XAUUSDT · 15m tìm Entry / 1H xác nhận / D1 xu hướng · "
                    "nến đóng và volume thật."
                ),
            )
        except Exception:
            logger.exception("Could not render/send peak 1H chart")
        await update.message.reply_text(
            reply,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        execution_guide = format_peak_execution_guide(
            gate,
            execution_plan,
            execution_reason,
            sized_plan,
            liquidity,
            hourly_structure,
            momentum_scores["1h"],
            daily_structure,
            momentum_scores["1day"],
            opportunity.quality,
            opportunity.trap,
            opportunity.macro_risk,
        )
        ai_execution_plan = execution_plan if sized_plan is not None else None
        ai_execution_reason = execution_reason
        if execution_plan is not None and sized_plan is None:
            ai_execution_reason += " Không tính được khối lượng an toàn nên không được đặt lệnh."

        ai_config = config.get("ai_analysis", {})
        if ai_config.get("enabled", True) and ai_config.get(
            "peak_review_enabled",
            True,
        ):
            snapshot = build_peak_ai_snapshot(
                peak_map=peak_map,
                gate=gate,
                frames=frames,
                momentum_scores=momentum_scores,
                derivatives_metrics={
                    "last_or_mid_price": realtime_quote.price,
                    "mark_price": realtime_quote.mark_price,
                    "index_price": realtime_quote.index_price,
                    "bid": realtime_quote.bid_price,
                    "ask": realtime_quote.ask_price,
                    "funding_rate": realtime_quote.funding_rate,
                    "open_interest": realtime_quote.open_interest,
                },
                execution_plan=ai_execution_plan,
                execution_reason=ai_execution_reason,
                liquidity=liquidity,
                hourly_structure=hourly_structure,
                daily_structure=daily_structure,
                setup_quality=opportunity.quality,
                liquidity_trap=opportunity.trap,
                macro_risk=opportunity.macro_risk,
            )
            timeout_seconds = int(ai_config.get("timeout_seconds", 35))
            candidates = []
            groq_key = os.getenv("GROQ_API_KEY")
            groq_config = ai_config.get("groq", {})
            if groq_key:
                groq_model = groq_config.get("model", "qwen/qwen3.6-27b")
                candidates.append(
                    {
                        "provider": "Groq",
                        "model": groq_model,
                        "label": f"Groq · {groq_model}",
                        "api_key": groq_key,
                        "analyzer": analyze_peak_with_groq,
                        "usage_path": groq_config.get(
                            "usage_path",
                            "logs/groq_usage.json",
                        ),
                        "daily_budget": int(
                            groq_config.get("daily_call_budget", 900)
                        ),
                        "cooldown_minutes": int(
                            groq_config.get("rate_limit_cooldown_minutes", 2)
                        ),
                    }
                )

            gemini_key = os.getenv("GEMINI_API_KEY")
            gemini_config = ai_config.get("gemini", {})
            if gemini_key:
                gemini_models = [
                    gemini_config.get("model", "gemini-3.6-flash"),
                    gemini_config.get(
                        "fallback_model",
                        "gemini-3.5-flash-lite",
                    ),
                ]
                for gemini_model in dict.fromkeys(filter(None, gemini_models)):
                    candidates.append(
                        {
                            "provider": "Gemini",
                            "model": gemini_model,
                            "label": f"Gemini · {gemini_model}",
                            "api_key": gemini_key,
                            "analyzer": analyze_peak_with_gemini,
                            "usage_path": gemini_config.get(
                                "usage_path",
                                "logs/gemini_usage.json",
                            ),
                            "daily_budget": int(
                                gemini_config.get("daily_call_budget", 15)
                            ),
                            "cooldown_minutes": int(
                                gemini_config.get(
                                    "rate_limit_cooldown_minutes",
                                    60,
                                )
                            ),
                        }
                    )

            if not candidates:
                await update.message.reply_text(
                    execution_guide
                    + "\n\n🤖 AI review đỉnh chưa được cấu hình; kế hoạch code phía trên vẫn dùng được."
                )
            else:
                review = None
                used_label = None
                last_error = None
                unavailable_reasons = []
                for candidate in candidates:
                    blocker_key = (
                        f"ai_blocked_until:{candidate['provider']}:"
                        f"{candidate['model']}"
                    )
                    blocked_until = context.bot_data.get(blocker_key)
                    if (
                        blocked_until is not None
                        and datetime.now(timezone.utc) < blocked_until
                    ):
                        unavailable_reasons.append(
                            f"{candidate['label']} đang chờ rate limit"
                        )
                        continue
                    if not ai_daily_budget_available(
                        candidate["usage_path"],
                        candidate["daily_budget"],
                    ):
                        unavailable_reasons.append(
                            f"{candidate['label']} đã hết ngân sách ngày"
                        )
                        continue
                    record_ai_call(candidate["usage_path"])
                    try:
                        review = await asyncio.wait_for(
                            asyncio.to_thread(
                                candidate["analyzer"],
                                candidate["api_key"],
                                candidate["model"],
                                timeout_seconds,
                                snapshot,
                            ),
                            timeout=timeout_seconds + 5,
                        )
                        used_label = candidate["label"]
                        break
                    except Exception as exc:
                        last_error = exc
                        if is_ai_rate_limit_error(exc):
                            context.bot_data[blocker_key] = (
                                datetime.now(timezone.utc)
                                + timedelta(
                                    minutes=candidate["cooldown_minutes"]
                                )
                            )
                            logger.warning(
                                "Peak AI %s rate-limited; trying fallback",
                                candidate["label"],
                            )
                            continue
                        logger.warning(
                            "Peak AI %s failed: %s",
                            candidate["label"],
                            type(exc).__name__,
                        )
                        continue

                if review is not None and used_label is not None:
                    await update.message.reply_text(
                        execution_guide
                        + "\n\n"
                        + format_peak_ai_review(
                            review,
                            used_label,
                            gate,
                        )
                    )
                elif last_error and is_ai_rate_limit_error(last_error):
                    await update.message.reply_text(
                        execution_guide
                        + "\n\n🤖 AI review đỉnh đang bị rate limit; bot không dùng kết quả cũ."
                    )
                elif unavailable_reasons:
                    await update.message.reply_text(
                        execution_guide
                        + "\n\n🤖 AI review: "
                        + "; ".join(unavailable_reasons)
                        + "."
                    )
                else:
                    await update.message.reply_text(
                        execution_guide
                        + "\n\n🤖 AI review đỉnh tạm lỗi; kế hoạch code phía trên vẫn là dữ liệu mới."
                    )
        else:
            await update.message.reply_text(execution_guide)
    except Exception:
        logger.exception("Failed to compute peak map")
        await update.message.reply_text(
            "Có lỗi khi quét đỉnh đa khung; thử lại /dinh sau ít phút."
        )


def _load_auto_alert_state(path: str) -> dict:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _save_auto_alert_state(path: str, state: dict) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _persist_active_wave(settings: dict, active_wave: dict | None) -> None:
    state_path = settings.get("state_path", "logs/auto_alert_state.json")
    state = _load_auto_alert_state(state_path)
    if active_wave is None:
        state.pop("active_wave", None)
    else:
        state["active_wave"] = active_wave
    _save_auto_alert_state(state_path, state)


def _parse_utc(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


MANUAL_POSITION_TEXT_PATTERN = re.compile(
    r"^/?(?P<side>long|short|sort)(?:@\w+)?:\s*"
    r"(?P<margin>\d+(?:[.,]\d+)?)\s*[_\s-]+\s*"
    r"(?P<leverage>\d+)\s*x?\s*$",
    re.IGNORECASE,
)
MANUAL_POSITION_COMMAND_PATTERN = re.compile(
    r"^/(?P<side>long|short|sort)(?:@\w+)?\s+"
    r"(?P<margin>\d+(?:[.,]\d+)?)\s*[_\s-]+\s*"
    r"(?P<leverage>\d+)\s*x?\s*$",
    re.IGNORECASE,
)


def parse_manual_position_command(text: str) -> tuple[str, float, int] | None:
    """Parse both Telegram-safe commands and the requested colon shorthand."""
    value = (text or "").strip()
    match = MANUAL_POSITION_TEXT_PATTERN.fullmatch(value)
    if match is None:
        match = MANUAL_POSITION_COMMAND_PATTERN.fullmatch(value)
    if match is None:
        return None
    side_token = match.group("side").lower()
    side = "LONG" if side_token == "long" else "SHORT"
    margin = float(match.group("margin").replace(",", "."))
    leverage = int(match.group("leverage"))
    return side, margin, leverage


def _manual_position_settings(config: dict) -> dict:
    return {
        **config.get("trading_plan", {}),
        **config.get("manual_position", {}),
    }


def _get_manual_position(context, settings: dict) -> dict | None:
    if "manual_position" in context.bot_data:
        value = context.bot_data["manual_position"]
        return value if isinstance(value, dict) else None
    state = _load_auto_alert_state(
        settings.get("state_path", "logs/manual_position_state.json")
    )
    value = state.get("position")
    position = value if isinstance(value, dict) else None
    context.bot_data["manual_position"] = position
    return position


def _set_manual_position(context, settings: dict, position: dict | None) -> None:
    context.bot_data["manual_position"] = position
    path = settings.get("state_path", "logs/manual_position_state.json")
    state = _load_auto_alert_state(path)
    if position is None:
        state.pop("position", None)
    else:
        state["position"] = position
    _save_auto_alert_state(path, state)


def build_manual_position_state(
    side: str,
    margin_usdt: float,
    leverage: int,
    entry_price: float,
    atr: float,
    opened_at: datetime,
    settings: dict,
) -> dict:
    """Create a price-monitoring plan; this does not represent an exchange fill."""
    if side not in {"LONG", "SHORT"}:
        raise ValueError("Manual position side must be LONG or SHORT")
    if margin_usdt <= 0 or leverage <= 0 or entry_price <= 0:
        raise ValueError("Margin, leverage and entry price must be positive")
    direction = 1.0 if side == "LONG" else -1.0
    price_tick = max(0.0000001, float(settings.get("price_tick", 0.01)))
    minimum_stop = entry_price * max(
        0.0,
        float(settings.get("minimum_stop_distance_pct", 0.05)),
    ) / 100
    maximum_stop = max(
        price_tick,
        float(settings.get("maximum_stop_distance", 7.0)),
    )
    raw_stop = max(
        price_tick,
        float(atr) * max(0.1, float(settings.get("stop_atr_multiplier", 0.80))),
        minimum_stop,
    )
    stop_distance = min(maximum_stop, raw_stop)
    tp1_r = max(0.1, float(settings.get("take_profit_1_r", 1.0)))
    tp2_r = max(tp1_r, float(settings.get("take_profit_2_r", 2.0)))

    def price_level(distance: float) -> float:
        return round((entry_price + direction * distance) / price_tick) * price_tick

    requested_notional = margin_usdt * leverage
    quantity_step = max(0.0000001, float(settings.get("quantity_step", 0.001)))
    minimum_quantity = max(quantity_step, float(settings.get("minimum_quantity", quantity_step)))
    quantity = math.floor((requested_notional / entry_price) / quantity_step) * quantity_step
    notional = quantity * entry_price
    minimum_notional = max(0.0, float(settings.get("minimum_notional_usdt", 5.0)))
    if quantity < minimum_quantity or notional < minimum_notional:
        raise ValueError(
            f"Ký quỹ × đòn bẩy quá nhỏ: cần notional tối thiểu {minimum_notional:.2f} USDT "
            f"và quantity tối thiểu {minimum_quantity:g} XAU"
        )
    fee_pct = max(0.0, float(settings.get("estimated_round_trip_fee_pct", 0.10)))
    slippage_bps = max(0.0, float(settings.get("estimated_slippage_bps", 2.0)))
    estimated_cost = notional * (fee_pct / 100 + slippage_bps / 10_000)
    stop_loss = price_level(-stop_distance)
    take_profit_1 = price_level(stop_distance * tp1_r)
    take_profit_2 = price_level(stop_distance * tp2_r)
    projected_stop_loss = quantity * stop_distance + estimated_cost
    if opened_at.tzinfo is None:
        opened_at = opened_at.replace(tzinfo=timezone.utc)
    else:
        opened_at = opened_at.astimezone(timezone.utc)
    return {
        "side": side,
        "margin_usdt": margin_usdt,
        "leverage": leverage,
        "notional_usdt": notional,
        "quantity_xau": quantity,
        "entry_price": entry_price,
        "entry_tracking_mode": "REFERENCE_QUOTE_NOT_EXCHANGE_FILL",
        "opened_at": opened_at.isoformat(),
        "risk_distance": stop_distance,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "estimated_round_trip_cost_usdt": estimated_cost,
        "projected_stop_loss_usdt": projected_stop_loss,
        "projected_stop_roe_pct": projected_stop_loss / margin_usdt * 100,
        "tp1_reached": False,
        "close_required_reason": None,
        "close_triggered_at": None,
        "last_alert_event": None,
        "last_alert_at": None,
        "last_check_at": opened_at.isoformat(),
        "last_price": entry_price,
    }


def manual_position_metrics(position: dict, exit_price: float) -> dict:
    side = position["side"]
    direction = 1.0 if side == "LONG" else -1.0
    entry = float(position["entry_price"])
    quantity = float(position["quantity_xau"])
    margin = max(1e-9, float(position["margin_usdt"]))
    risk_distance = max(1e-9, float(position["risk_distance"]))
    gross_pnl = direction * (exit_price - entry) * quantity
    estimated_net_pnl = gross_pnl - float(
        position.get("estimated_round_trip_cost_usdt", 0.0)
    )
    return {
        "exit_price": exit_price,
        "gross_pnl_usdt": gross_pnl,
        "estimated_net_pnl_usdt": estimated_net_pnl,
        "roe_pct": estimated_net_pnl / margin * 100,
        "favorable_move_pct": direction * (exit_price - entry) / entry * 100,
        "r_multiple": direction * (exit_price - entry) / risk_distance,
    }


def update_manual_position_event(
    position: dict,
    metrics: dict,
    checked_at: datetime,
    settings: dict,
) -> str:
    """Latch hard exit events so reminders continue until the user sends /dong."""
    if position.get("close_required_reason"):
        return "CLOSE_REQUIRED"
    side = position["side"]
    price = float(metrics["exit_price"])
    stop = float(position["stop_loss"])
    tp1 = float(position["take_profit_1"])
    tp2 = float(position["take_profit_2"])
    stop_hit = price <= stop if side == "LONG" else price >= stop
    tp2_hit = price >= tp2 if side == "LONG" else price <= tp2
    tp1_hit = price >= tp1 if side == "LONG" else price <= tp1
    if stop_hit or tp2_hit:
        position["close_required_reason"] = "STOP_LOSS" if stop_hit else "TAKE_PROFIT_2"
        position["close_triggered_at"] = checked_at.astimezone(timezone.utc).isoformat()
        position["close_trigger_price"] = price
        return "CLOSE_REQUIRED"
    if tp1_hit:
        position["tp1_reached"] = True
        return "TAKE_PROFIT_1"
    if position.get("tp1_reached"):
        entry = float(position["entry_price"])
        breakeven_hit = price <= entry if side == "LONG" else price >= entry
        if breakeven_hit:
            position["close_required_reason"] = "BREAKEVEN_AFTER_TP1"
            position["close_triggered_at"] = checked_at.astimezone(timezone.utc).isoformat()
            position["close_trigger_price"] = price
            return "CLOSE_REQUIRED"
        return "TAKE_PROFIT_1"
    r_multiple = float(metrics["r_multiple"])
    if r_multiple <= -max(0.1, float(settings.get("loss_warning_r", 0.50))):
        return "LOSS_WARNING"
    if r_multiple >= max(0.1, float(settings.get("profit_reminder_r", 0.50))):
        return "PROFIT_REMINDER"
    return "STATUS"


def _manual_exit_price(position: dict, quote: RealtimeQuote) -> float:
    if position["side"] == "LONG" and quote.bid_price is not None:
        return float(quote.bid_price)
    if position["side"] == "SHORT" and quote.ask_price is not None:
        return float(quote.ask_price)
    return float(quote.price)


def _manual_alert_interval(event: str, settings: dict) -> int:
    if event == "CLOSE_REQUIRED":
        return max(10, int(settings.get("close_repeat_seconds", 30)))
    if event == "STATUS":
        return max(30, int(settings.get("status_interval_seconds", 300)))
    return max(30, int(settings.get("reminder_interval_seconds", 180)))


def manual_alert_is_due(
    position: dict,
    event: str,
    checked_at: datetime,
    settings: dict,
) -> bool:
    if position.get("last_alert_event") != event:
        return True
    last_alert = _parse_utc(position.get("last_alert_at"))
    if last_alert is None:
        return True
    return (checked_at - last_alert).total_seconds() >= _manual_alert_interval(
        event,
        settings,
    )


def format_manual_position_alert(
    position: dict,
    metrics: dict,
    event: str,
    checked_at: datetime,
    settings: dict,
) -> str:
    side = position["side"]
    reason = position.get("close_required_reason")
    if event == "CLOSE_REQUIRED":
        reason_text = {
            "STOP_LOSS": "giá đã chạm/vượt mức cắt lỗ",
            "TAKE_PROFIT_2": "giá đã chạm TP2 — nên khóa lợi nhuận",
            "BREAKEVEN_AFTER_TP1": "sau TP1 giá đã quay về Entry",
        }.get(reason, "điều kiện đóng vị thế đã kích hoạt")
        title = f"🚨🚨 CẦN ĐÓNG {side} — {reason_text.upper()}"
        action = (
            "• Nếu lệnh thật vẫn còn mở: kiểm tra Binance và đóng/giữ đúng lệnh bảo vệ ngay. "
            f"Bot sẽ nhắc lại mỗi {_manual_alert_interval(event, settings)} giây cho đến khi bạn gửi /dong.\n"
            "• Bot không có quyền đóng lệnh và không biết lệnh thật đã fill/đóng hay chưa."
        )
    elif event == "TAKE_PROFIT_1":
        title = f"✅ {side} ĐÃ CHẠM TP1 — NHẮC CHỐT LÃI"
        action = (
            "• Nếu chưa làm: cân nhắc chốt 50% và dời SL phần còn lại về Entry. "
            "Bot tiếp tục theo dõi cho đến /dong."
        )
    elif event == "LOSS_WARNING":
        title = f"⚠️ {side} ĐANG TIẾN GẦN SL"
        action = "• Không thêm vị thế/gồng lỗ; kiểm tra Stop-Market thật đang hoạt động."
    elif event == "PROFIT_REMINDER":
        title = f"📈 {side} ĐANG CÓ LÃI"
        action = "• Chưa chạm TP1; không nới SL và không tăng size vì FOMO."
    else:
        title = f"🔎 CẬP NHẬT VỊ THẾ {side}"
        action = (
            f"• Bot vẫn kiểm tra mỗi {int(settings.get('poll_seconds', 10))} giây; "
            "gửi /dong sau khi bạn đã đóng lệnh thật."
        )

    return "\n".join(
        [
            title,
            f"• Entry tham chiếu {float(position['entry_price']):.2f} → giá thoát tham chiếu {float(metrics['exit_price']):.2f} · {float(metrics['r_multiple']):+.2f}R.",
            f"• PnL gộp {float(metrics['gross_pnl_usdt']):+.2f} USDT · ước tính sau phí/trượt {float(metrics['estimated_net_pnl_usdt']):+.2f} USDT · ROE {float(metrics['roe_pct']):+.2f}%.",
            f"• Ký quỹ {float(position['margin_usdt']):.2f} USDT · {int(position['leverage'])}x · notional {float(position['notional_usdt']):.2f} USDT · {float(position['quantity_xau']):.3f} XAU.",
            f"• SL {float(position['stop_loss']):.2f} · TP1 {float(position['take_profit_1']):.2f} · TP2 {float(position['take_profit_2']):.2f}.",
            action,
            f"• Kiểm tra lúc {checked_at.astimezone().strftime('%d/%m %H:%M:%S')}.",
        ]
    )


def _capture_manual_position_market(
    config: dict,
    provider,
    price_stream,
    side: str,
) -> tuple[RealtimeQuote, float, float, datetime]:
    checked_at = datetime.now(timezone.utc)
    quote = get_current_quote(provider, price_stream)
    settings = _manual_position_settings(config)
    bars = max(40, min(1499, int(settings.get("atr_bars", 100))))
    frame = select_closed_candles(
        provider.get_historical("15min", outputsize=bars + 1),
        "15min",
        checked_at,
    ).tail(bars)
    signal = compute_signal(
        frame,
        config["weights"],
        {"buy": config["threshold_buy"], "sell": config["threshold_sell"]},
        atr_stop_multiplier=config.get("atr_stop_multiplier", 1.5),
    )
    entry_price = (
        float(quote.ask_price)
        if side == "LONG" and quote.ask_price is not None
        else float(quote.bid_price)
        if side == "SHORT" and quote.bid_price is not None
        else float(quote.price)
    )
    return quote, entry_price, float(signal.atr), checked_at


def _auto_alert_fingerprint(plan: PeakExecutionPlan, phase: str) -> str:
    return "|".join(
        [
            plan.side,
            f"{plan.entry_lower:.2f}",
            f"{plan.entry_upper:.2f}",
            f"{plan.stop_loss:.2f}",
            f"{plan.take_profit_1:.2f}",
            f"{plan.take_profit_2:.2f}",
            phase,
        ]
    )


def _auto_alert_phase(
    opportunity: PeakOpportunity,
    settings: dict,
) -> tuple[str | None, float]:
    plan = opportunity.execution_plan
    if plan is None or opportunity.sized_plan is None:
        return None, opportunity.realtime_quote.price
    quote = opportunity.realtime_quote
    executable_price = (
        quote.ask_price
        if plan.side == "LONG" and quote.ask_price is not None
        else quote.bid_price
        if plan.side == "SHORT" and quote.bid_price is not None
        else quote.price
    )
    if plan.entry_lower <= executable_price <= plan.entry_upper:
        return "IN_ZONE", executable_price
    distance = min(
        abs(executable_price - plan.entry_lower),
        abs(executable_price - plan.entry_upper),
    )
    approach_pct = max(0.0, float(settings.get("approach_buffer_pct", 0.01)))
    if settings.get("notify_approaching", True) and distance <= quote.price * approach_pct / 100:
        return "APPROACHING", executable_price
    return None, executable_price


def _record_auto_analysis(
    context,
    opportunity: PeakOpportunity,
    phase: str | None,
    executable_price: float,
    event: str = "analysis_snapshot",
    force: bool = False,
) -> None:
    config = context.bot_data["config"]
    settings = config.get("analysis_diagnostics", {})
    if not settings.get("enabled", True):
        return
    plan = opportunity.execution_plan
    latest_15m = opportunity.frames["15min"].index[-1].isoformat()
    signature = (
        latest_15m,
        opportunity.gate.allowed_decision,
        opportunity.gate.reason,
        opportunity.execution_reason,
        opportunity.quality.score,
        opportunity.quality.actionable,
        opportunity.macro_risk.blocked,
        opportunity.trap.double_sweep,
        opportunity.trap.fomo_extension,
        phase,
        (
            plan.side,
            plan.entry_lower,
            plan.entry_upper,
            plan.stop_loss,
            plan.take_profit_1,
            plan.take_profit_2,
        )
        if plan is not None
        else None,
    )
    signature_key = f"last_analysis_diagnostic_signature:{event}"
    if not force and context.bot_data.get(signature_key) == signature:
        return
    try:
        record = build_analysis_snapshot(
            opportunity,
            config,
            phase,
            executable_price,
            event=event,
        )
        path = append_analysis_snapshot(record, settings)
        context.bot_data[signature_key] = signature
        context.bot_data["last_analysis_diagnostic_path"] = str(path) if path else None
    except Exception:
        logger.exception("Failed to write analysis diagnostic snapshot")


def _record_diagnostic_event(
    context,
    event: str,
    details: dict,
    occurred_at: datetime | None = None,
) -> None:
    config = context.bot_data["config"]
    settings = config.get("analysis_diagnostics", {})
    if not settings.get("enabled", True):
        return
    try:
        record = build_diagnostic_event(
            event,
            occurred_at or datetime.now(timezone.utc),
            details,
            symbol=config.get("symbol", "XAUUSDT"),
        )
        path = append_analysis_snapshot(record, settings)
        context.bot_data["last_analysis_diagnostic_path"] = str(path) if path else None
    except Exception:
        logger.exception("Failed to write diagnostic event: %s", event)


def _build_active_wave(
    opportunity: PeakOpportunity,
    phase: str,
    checked_at: datetime,
    settings: dict,
) -> dict:
    plan = opportunity.execution_plan
    sized_plan = opportunity.sized_plan
    if plan is None or sized_plan is None:
        raise ValueError("Active wave requires a complete execution and sizing plan")
    entered = phase == "IN_ZONE"
    timeout_minutes = (
        min(
            max(1, sized_plan.max_holding_minutes),
            max(1, int(settings.get("active_wave_max_minutes", 480))),
        )
        if entered
        else max(1, int(settings.get("setup_timeout_minutes", 90)))
    )
    return {
        "fingerprint": _auto_alert_fingerprint(plan, "TRACKING"),
        "side": plan.side,
        "phase": "ACTIVE" if entered else "WAITING_ENTRY",
        "entered": entered,
        "entry_tracking_mode": "SIMULATED_PRICE_TOUCH",
        "paper_only": opportunity.quality.paper_only,
        "started_at": checked_at.isoformat(),
        "entered_at": checked_at.isoformat() if entered else None,
        "expires_at": (checked_at + timedelta(minutes=timeout_minutes)).isoformat(),
        "last_check_at": checked_at.isoformat(),
        "last_price": opportunity.realtime_quote.price,
        "entry_lower": plan.entry_lower,
        "entry_upper": plan.entry_upper,
        "entry_reference": plan.entry_reference,
        "stop_loss": plan.stop_loss,
        "take_profit_1": plan.take_profit_1,
        "take_profit_2": plan.take_profit_2,
        "quantity_xau": sized_plan.quantity_xau,
        "leverage": sized_plan.leverage,
        "risk_usdt": sized_plan.risk_usdt,
        "max_holding_minutes": sized_plan.max_holding_minutes,
        "tp1_notified": False,
        "weakness_notified": False,
        "last_micro_bucket": None,
        "pressure_score": None,
        "pressure_label": None,
        "micro_monotonic_opposite": False,
        "last_micro_close": None,
        "last_micro_closed_at": None,
    }


def _get_active_wave(context, settings: dict) -> dict | None:
    if "active_wave" in context.bot_data:
        value = context.bot_data["active_wave"]
        return value if isinstance(value, dict) else None
    state = _load_auto_alert_state(
        settings.get("state_path", "logs/auto_alert_state.json")
    )
    value = state.get("active_wave")
    active_wave = value if isinstance(value, dict) else None
    if active_wave is not None:
        expires_at = _parse_utc(active_wave.get("expires_at"))
        started_at = _parse_utc(active_wave.get("started_at"))
        too_stale = (
            started_at is None
            or datetime.now(timezone.utc) - started_at > timedelta(hours=24)
        )
        if expires_at is None or too_stale:
            active_wave = None
            _persist_active_wave(settings, None)
    context.bot_data["active_wave"] = active_wave
    return active_wave


def _set_active_wave(context, settings: dict, active_wave: dict | None) -> None:
    context.bot_data["active_wave"] = active_wave
    _persist_active_wave(settings, active_wave)


def _wave_price(active_wave: dict, quote: RealtimeQuote, entering: bool) -> float:
    side = active_wave["side"]
    if entering:
        if side == "LONG" and quote.ask_price is not None:
            return quote.ask_price
        if side == "SHORT" and quote.bid_price is not None:
            return quote.bid_price
    else:
        if side == "LONG" and quote.bid_price is not None:
            return quote.bid_price
        if side == "SHORT" and quote.ask_price is not None:
            return quote.ask_price
    return quote.price


def _wave_r(active_wave: dict) -> float:
    return max(
        1e-9,
        abs(float(active_wave["entry_reference"]) - float(active_wave["stop_loss"])),
    )


def _wave_progress_r(active_wave: dict, price: float) -> float:
    direction = 1.0 if active_wave["side"] == "LONG" else -1.0
    return direction * (price - float(active_wave["entry_reference"])) / _wave_r(active_wave)


def _format_active_wave_alert(
    event: str,
    active_wave: dict,
    price: float,
    detail: str = "",
    active_poll_seconds: int = 10,
) -> str:
    side = active_wave["side"]
    entry_lower = float(active_wave["entry_lower"])
    entry_upper = float(active_wave["entry_upper"])
    stop = float(active_wave["stop_loss"])
    tp1 = float(active_wave["take_profit_1"])
    tp2 = float(active_wave["take_profit_2"])
    paper_only = bool(active_wave.get("paper_only", True))
    paper_label = "🧪 PAPER ONLY · " if paper_only else ""
    progress = _wave_progress_r(active_wave, price)
    checked = datetime.now().astimezone().strftime("%H:%M:%S")
    if event == "ENTRY":
        return (
            f"🚨 {paper_label}{side} ĐÃ CHẠM ENTRY (KHỚP GIẢ ĐỊNH)\n"
            f"• Giá thực thi {price:.2f} đã vào Entry {entry_lower:.2f}–{entry_upper:.2f}.\n"
            f"• SL {stop:.2f} · TP1 {tp1:.2f} · TP2 {tp2:.2f}.\n"
            + (
                "• Ghi fill giả lập vào journal; không đặt lệnh thật. "
                if paper_only
                else "• Hãy kiểm tra lệnh thật có fill hay không; bot không đọc tài khoản. "
            )
            + f"Bot bắt đầu mô phỏng mỗi {active_poll_seconds} giây ({checked})."
        )
    if event == "WEAK":
        return (
            f"⚠️ {paper_label}SÓNG {side} ĐANG YẾU\n"
            f"• Giá {price:.2f} · tiến độ {progress:+.2f}R. {detail}\n"
            + (
                "• Không thêm paper entry; giữ mốc SL giả lập và chờ xác nhận 15m."
                if paper_only
                else "• Không thêm lệnh; giữ đúng SL và chờ xác nhận 15m. Chưa coi là kết thúc sóng."
            )
        )
    if event == "RETEST_FAILED":
        return (
            f"🛑🛑 {paper_label}RETEST {side} THẤT BẠI — CẮT NGAY\n"
            f"• Giá hiện tại {price:.2f}. {detail}\n"
            + (
                "• Ghi đóng toàn bộ vị thế giả lập trong journal; không thao tác lệnh thật.\n"
                if paper_only
                else "• Đóng TOÀN BỘ bằng Market Close/Close Position và Reduce-Only; hủy TP/SL còn treo.\n"
            )
            + "• Không chờ hard SL, không gồng và không tự động đảo chiều."
        )
    if event == "TP1":
        return (
            f"✅ {paper_label}{side} ĐÃ CHẠM TP1 {tp1:.2f}\n"
            f"• Giá {price:.2f} · tiến độ {progress:+.2f}R.\n"
            + (
                f"• Ghi chốt giả lập 50% và dời SL giả lập về Entry {float(active_wave['entry_reference']):.2f}."
                if paper_only
                else f"• Chốt 50% và dời SL phần còn lại về Entry {float(active_wave['entry_reference']):.2f}."
            )
        )
    if event == "TP2":
        return (
            f"🛑🛑 {paper_label}SÓNG {side} ĐÃ HOÀN TẤT — CHỐT LỆNH\n"
            f"• Giá {price:.2f} đã chạm TP2 {tp2:.2f} ({progress:+.2f}R).\n"
            + (
                "• Ghi chốt phần giả lập còn lại; không đặt lệnh thật."
                if paper_only
                else "• Chốt phần còn lại, hủy lệnh chờ và không đuổi theo sóng cũ."
            )
        )
    if event == "BREAKEVEN":
        return (
            f"🛑🛑 {paper_label}SÓNG {side} ĐÃ KẾT THÚC — BẢO TOÀN LỢI NHUẬN\n"
            f"• Sau TP1, giá đã quay về Entry {float(active_wave['entry_reference']):.2f}.\n"
            + (
                "• Ghi đóng phần giả lập còn lại tại hòa vốn; không vào lệnh thật."
                if paper_only
                else "• Đóng phần còn lại theo SL hòa vốn; không gồng và không vào lại ngay."
            )
        )
    if event in {"STOP", "STRUCTURE_END"}:
        reason = (
            f"giá đã chạm mức vô hiệu/SL {stop:.2f}"
            if event == "STOP"
            else detail
        )
        return (
            f"🛑🛑 {paper_label}SÓNG {side} ĐÃ KẾT THÚC — DỪNG KÈO\n"
            f"• Giá {price:.2f}: {reason}.\n"
            + (
                "• Ghi thoát paper trade theo SL; không thao tác lệnh thật."
                if paper_only
                else "• Đóng/giữ đúng SL, không gồng lỗ và không tự động đảo chiều."
            )
        )
    if event == "TIME_STOP":
        return (
            f"🛑🛑 {paper_label}SÓNG {side} ĐÃ HẾT THỜI GIAN — ĐÓNG KÈO\n"
            f"• Giá hiện tại {price:.2f} · tiến độ {progress:+.2f}R.\n"
            + (
                "• Ghi đóng paper trade theo time-stop; không đặt lệnh thật."
                if paper_only
                else "• Đóng phần còn lại theo time-stop; không kéo dài một kèo ngắn hạn sang qua ngày."
            )
        )
    if event in {"INVALIDATED", "RUNAWAY", "EXPIRED"}:
        title = {
            "INVALIDATED": "SETUP ĐÃ BỊ VÔ HIỆU",
            "RUNAWAY": "BỎ KÈO — KHÔNG ĐUỔI GIÁ",
            "EXPIRED": "SETUP ĐÃ HẾT HẠN",
        }[event]
        return (
            f"⛔ {title} ({side})\n"
            f"• Giá {price:.2f}. {detail}\n"
            "• Bot dừng canh setup này; chờ tín hiệu cấu trúc mới."
        )
    raise ValueError(f"Unsupported active-wave event: {event}")


def _fast_wave_event(active_wave: dict, price: float, settings: dict) -> str | None:
    side = active_wave["side"]
    entered = bool(active_wave.get("entered"))
    stop = float(active_wave["stop_loss"])
    tp1 = float(active_wave["take_profit_1"])
    tp2 = float(active_wave["take_profit_2"])
    entry = float(active_wave["entry_reference"])
    lower = float(active_wave["entry_lower"])
    upper = float(active_wave["entry_upper"])

    if not entered:
        if (side == "LONG" and price <= stop) or (side == "SHORT" and price >= stop):
            return "INVALIDATED"
        if lower <= price <= upper:
            return "ENTRY"
        runaway_r = max(0.1, float(settings.get("runaway_cancel_r", 0.5)))
        if (side == "LONG" and price > upper + runaway_r * _wave_r(active_wave)) or (
            side == "SHORT" and price < lower - runaway_r * _wave_r(active_wave)
        ):
            return "RUNAWAY"
        expires_at = _parse_utc(active_wave.get("expires_at"))
        if expires_at is not None and datetime.now(timezone.utc) >= expires_at:
            return "EXPIRED"
        return None

    if (side == "LONG" and price >= tp2) or (side == "SHORT" and price <= tp2):
        return "TP2"
    if active_wave.get("tp1_notified") and (
        (side == "LONG" and price <= entry) or (side == "SHORT" and price >= entry)
    ):
        return "BREAKEVEN"
    if (side == "LONG" and price <= stop) or (side == "SHORT" and price >= stop):
        return "STOP"
    if not active_wave.get("tp1_notified") and (
        (side == "LONG" and price >= tp1) or (side == "SHORT" and price <= tp1)
    ):
        return "TP1"
    expires_at = _parse_utc(active_wave.get("expires_at"))
    if expires_at is not None and datetime.now(timezone.utc) >= expires_at:
        return "TIME_STOP"
    return None


def _refresh_micro_pressure(
    provider,
    active_wave: dict,
    checked_at: datetime,
    settings: dict,
) -> tuple[float | None, bool, bool, float | None, datetime | None]:
    """Fetch closed 1m candles once per minute, not on every fast price tick."""
    current_bucket = checked_at.replace(second=0, microsecond=0).isoformat()
    if active_wave.get("last_micro_bucket") == current_bucket:
        return (
            active_wave.get("pressure_score"),
            bool(active_wave.get("micro_monotonic_opposite")),
            False,
            active_wave.get("last_micro_close"),
            _parse_utc(active_wave.get("last_micro_closed_at")),
        )
    bars = max(10, min(100, int(settings.get("micro_candle_bars", 30))))
    frame = select_closed_candles(
        provider.get_historical("1min", outputsize=bars + 1),
        "1min",
        checked_at,
    ).tail(bars)
    pressure = analyze_price_pressure(frame)
    closes = frame["close"].astype(float).tail(3)
    deltas = closes.diff().dropna()
    side = active_wave["side"]
    monotonic_opposite = bool(
        len(deltas) >= 2
        and ((deltas < 0).all() if side == "LONG" else (deltas > 0).all())
    )
    active_wave["last_micro_bucket"] = current_bucket
    active_wave["pressure_score"] = pressure.score
    active_wave["pressure_label"] = pressure.label
    active_wave["micro_monotonic_opposite"] = monotonic_opposite
    last_micro_close = float(frame["close"].iloc[-1])
    last_micro_closed_at = frame.index[-1].to_pydatetime() + timedelta(minutes=1)
    if last_micro_closed_at.tzinfo is None:
        last_micro_closed_at = last_micro_closed_at.replace(tzinfo=timezone.utc)
    else:
        last_micro_closed_at = last_micro_closed_at.astimezone(timezone.utc)
    active_wave["last_micro_close"] = last_micro_close
    active_wave["last_micro_closed_at"] = last_micro_closed_at.isoformat()
    return (
        pressure.score,
        monotonic_opposite,
        True,
        last_micro_close,
        last_micro_closed_at,
    )


def _micro_retest_failure(
    active_wave: dict,
    pressure_score: float | None,
    last_micro_close: float | None,
    last_micro_closed_at: datetime | None,
    settings: dict,
) -> tuple[bool, str]:
    """Confirm an early retest failure only with a post-entry closed 1m candle."""
    entered_at = _parse_utc(active_wave.get("entered_at"))
    if (
        entered_at is None
        or last_micro_close is None
        or last_micro_closed_at is None
        or last_micro_closed_at <= entered_at
        or pressure_score is None
    ):
        return False, ""
    buffer_price = max(
        0.0,
        float(settings.get("retest_failure_buffer_price", 0.50)),
        _wave_r(active_wave)
        * max(0.0, float(settings.get("retest_failure_buffer_r", 0.05))),
    )
    pressure_threshold = max(
        0.05,
        float(settings.get("retest_failure_pressure_threshold", 0.20)),
    )
    side = active_wave["side"]
    lower = float(active_wave["entry_lower"])
    upper = float(active_wave["entry_upper"])
    if (
        side == "LONG"
        and last_micro_close < lower - buffer_price
        and pressure_score <= -pressure_threshold
    ):
        return (
            True,
            f"Nến 1m đóng {last_micro_close:.2f} dưới vùng Entry "
            f"{lower:.2f} với áp lực bán {pressure_score:+.2f}.",
        )
    if (
        side == "SHORT"
        and last_micro_close > upper + buffer_price
        and pressure_score >= pressure_threshold
    ):
        return (
            True,
            f"Nến 1m đóng {last_micro_close:.2f} trên vùng Entry "
            f"{upper:.2f} với áp lực mua {pressure_score:+.2f}.",
        )
    return False, ""


def _full_scan_wave_end_event(
    active_wave: dict,
    opportunity: PeakOpportunity,
    settings: dict,
) -> tuple[str | None, str]:
    if not active_wave.get("entered"):
        return None, ""
    side = active_wave["side"]
    stop = float(active_wave["stop_loss"])
    last_15m_close = float(opportunity.frames["15min"]["close"].iloc[-1])
    if (side == "LONG" and last_15m_close <= stop) or (
        side == "SHORT" and last_15m_close >= stop
    ):
        return "STRUCTURE_END", "nến 15m đã đóng phá mức vô hiệu"
    threshold = max(0.05, float(settings.get("structure_flip_threshold", 0.15)))
    score_15m = opportunity.momentum_scores["15min"]
    score_1h = opportunity.momentum_scores["1h"]
    both_opposite = (
        score_15m <= -threshold and score_1h <= -threshold
        if side == "LONG"
        else score_15m >= threshold and score_1h >= threshold
    )
    if both_opposite:
        return (
            "STRUCTURE_END",
            f"15m và 1H đã cùng đảo chiều (điểm {score_15m:+.2f}/{score_1h:+.2f})",
        )
    return None, ""


def format_auto_entry_alert(
    opportunity: PeakOpportunity,
    phase: str,
    executable_price: float,
) -> str:
    plan = opportunity.execution_plan
    sized_plan = opportunity.sized_plan
    if plan is None or sized_plan is None:
        raise ValueError("Auto alert requires a complete execution and sizing plan")
    phase_text = (
        "GIÁ ĐÃ CHẠM ENTRY · KHỚP GIẢ ĐỊNH"
        if phase == "IN_ZONE"
        else "GIÁ ĐANG SÁT VÙNG ENTRY"
    )
    checked_at = opportunity.analysis_now.astimezone().strftime("%d/%m %H:%M:%S")
    guide = format_peak_execution_guide(
        opportunity.gate,
        plan,
        opportunity.execution_reason,
        sized_plan,
        opportunity.liquidity,
        opportunity.hourly_structure,
        opportunity.momentum_scores["1h"],
        opportunity.daily_structure,
        opportunity.momentum_scores["1day"],
        opportunity.quality,
        opportunity.trap,
        opportunity.macro_risk,
    )
    return "\n".join(
        [
            f"🔔 AUTO XAUUSDT — {phase_text}"
            + (" · PAPER ONLY" if opportunity.quality.paper_only else ""),
            f"• {plan.side} · giá thực thi tham chiếu {executable_price:.2f} · kiểm tra {checked_at}.",
            (
                "• Chỉ ghi journal giả lập; không đặt lệnh thật. Bot mô phỏng chạm giá."
                if opportunity.quality.paper_only
                else "• Mở Binance để xác nhận lệnh thật có fill; bot chỉ mô phỏng chạm giá và không tự đặt lệnh."
            ),
            "",
            guide,
        ]
    )


def _build_auto_peak_ai_snapshot(opportunity: PeakOpportunity) -> dict:
    return build_peak_ai_snapshot(
        peak_map=opportunity.peak_map,
        gate=opportunity.gate,
        frames=opportunity.frames,
        momentum_scores=opportunity.momentum_scores,
        derivatives_metrics={
            "last_or_mid_price": opportunity.realtime_quote.price,
            "mark_price": opportunity.realtime_quote.mark_price,
            "index_price": opportunity.realtime_quote.index_price,
            "bid": opportunity.realtime_quote.bid_price,
            "ask": opportunity.realtime_quote.ask_price,
            "funding_rate": opportunity.realtime_quote.funding_rate,
            "open_interest": opportunity.realtime_quote.open_interest,
        },
        execution_plan=opportunity.execution_plan,
        execution_reason=opportunity.execution_reason,
        liquidity=opportunity.liquidity,
        hourly_structure=opportunity.hourly_structure,
        daily_structure=opportunity.daily_structure,
        setup_quality=opportunity.quality,
        liquidity_trap=opportunity.trap,
        macro_risk=opportunity.macro_risk,
    )


def _auto_peak_ai_candidates(ai_config: dict) -> list[dict]:
    candidates = []
    groq_key = os.getenv("GROQ_API_KEY")
    groq_config = ai_config.get("groq", {})
    if groq_key:
        groq_model = groq_config.get("model", "qwen/qwen3.6-27b")
        candidates.append(
            {
                "provider": "Groq",
                "model": groq_model,
                "label": f"Groq · {groq_model}",
                "api_key": groq_key,
                "analyzer": analyze_peak_with_groq,
                "usage_path": groq_config.get(
                    "usage_path",
                    "logs/groq_usage.json",
                ),
                "daily_budget": int(groq_config.get("daily_call_budget", 900)),
                "cooldown_minutes": int(
                    groq_config.get("rate_limit_cooldown_minutes", 2)
                ),
            }
        )

    gemini_key = os.getenv("GEMINI_API_KEY")
    gemini_config = ai_config.get("gemini", {})
    if gemini_key:
        gemini_models = [
            gemini_config.get("model", "gemini-3.6-flash"),
            gemini_config.get("fallback_model", "gemini-3.5-flash-lite"),
        ]
        for gemini_model in dict.fromkeys(filter(None, gemini_models)):
            candidates.append(
                {
                    "provider": "Gemini",
                    "model": gemini_model,
                    "label": f"Gemini · {gemini_model}",
                    "api_key": gemini_key,
                    "analyzer": analyze_peak_with_gemini,
                    "usage_path": gemini_config.get(
                        "usage_path",
                        "logs/gemini_usage.json",
                    ),
                    "daily_budget": int(
                        gemini_config.get("daily_call_budget", 15)
                    ),
                    "cooldown_minutes": int(
                        gemini_config.get("rate_limit_cooldown_minutes", 60)
                    ),
                }
            )
    return candidates


async def _request_auto_peak_ai_review(
    context: ContextTypes.DEFAULT_TYPE,
    snapshot: dict,
) -> tuple[object | None, str | None, str | None]:
    ai_config = context.bot_data["config"].get("ai_analysis", {})
    if not ai_config.get("enabled", True) or not ai_config.get(
        "peak_review_enabled",
        True,
    ):
        return None, None, "AI review đang tắt trong cấu hình"
    candidates = _auto_peak_ai_candidates(ai_config)
    if not candidates:
        return None, None, "chưa có GROQ_API_KEY hoặc GEMINI_API_KEY"

    timeout_seconds = int(ai_config.get("timeout_seconds", 35))
    unavailable_reasons = []
    last_error = None
    for candidate in candidates:
        blocker_key = (
            f"ai_blocked_until:{candidate['provider']}:{candidate['model']}"
        )
        blocked_until = context.bot_data.get(blocker_key)
        if blocked_until is not None and datetime.now(timezone.utc) < blocked_until:
            unavailable_reasons.append(f"{candidate['label']} đang cooldown")
            continue
        if not ai_daily_budget_available(
            candidate["usage_path"],
            candidate["daily_budget"],
        ):
            unavailable_reasons.append(
                f"{candidate['label']} đã hết ngân sách ngày"
            )
            continue

        record_ai_call(candidate["usage_path"])
        try:
            review = await asyncio.wait_for(
                asyncio.to_thread(
                    candidate["analyzer"],
                    candidate["api_key"],
                    candidate["model"],
                    timeout_seconds,
                    snapshot,
                ),
                timeout=timeout_seconds + 5,
            )
            return review, candidate["label"], None
        except Exception as exc:
            last_error = exc
            if is_ai_rate_limit_error(exc):
                context.bot_data[blocker_key] = (
                    datetime.now(timezone.utc)
                    + timedelta(minutes=candidate["cooldown_minutes"])
                )
                unavailable_reasons.append(f"{candidate['label']} bị rate limit")
            else:
                unavailable_reasons.append(
                    f"{candidate['label']} lỗi {type(exc).__name__}"
                )
            logger.warning(
                "Auto alert AI %s failed: %s",
                candidate["label"],
                type(exc).__name__,
            )

    if unavailable_reasons:
        return None, None, "; ".join(unavailable_reasons)
    if last_error is not None:
        return None, None, f"AI lỗi {type(last_error).__name__}"
    return None, None, "không có model AI khả dụng"


async def auto_alert_ai_review_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the second, asynchronous AI verification message for an auto alert."""
    data = context.job.data if context.job is not None else None
    if not isinstance(data, dict):
        return
    chat_id = data["chat_id"]
    gate = data["gate"]
    side = data["side"]
    try:
        config = context.bot_data["config"]
        ai_config = config.get("ai_analysis", {})
        active_poll_seconds = int(
            config.get("auto_alerts", {}).get("active_poll_seconds", 10)
        )
        review, used_label, unavailable_reason = await asyncio.wait_for(
            _request_auto_peak_ai_review(
                context,
                data["snapshot"],
            ),
            timeout=max(
                10,
                int(ai_config.get("auto_review_total_timeout_seconds", 50)),
            ),
        )
        paper_only = bool(
            (data.get("snapshot", {}).get("setup_quality_score") or {}).get(
                "paper_only",
                True,
            )
        )
        expected_decision = f"CANH {side}"
        if review is not None and used_label is not None:
            if review.decision == expected_decision:
                verdict = (
                    f"✅ AI XÁC THỰC: ĐỒNG THUẬN {side}"
                    + (" · PAPER ONLY" if paper_only else "")
                    + "\nAI chỉ là lớp kiểm tra thứ hai, không phải bảo đảm thắng."
                )
            else:
                verdict = (
                    f"⚠️ AI CHƯA XÁC THỰC KÈO {side}\n"
                    + (
                        "Tiếp tục ghi paper journal; không đặt lệnh thật."
                        if paper_only
                        else "Nếu chưa vào: tiếp tục đứng ngoài. Nếu đã khớp: không tăng vị thế và giữ đúng SL."
                    )
                )
            message = (
                verdict
                + "\n\n"
                + format_peak_ai_review(review, used_label, gate)
            )
        else:
            message = (
                f"⚠️ AI XÁC THỰC {side} KHÔNG KHẢ DỤNG\n"
                f"• {unavailable_reason or 'AI không phản hồi'}.\n"
                f"• Không coi đây là AI đồng thuận; bot vẫn canh giá {active_poll_seconds} giây và giữ nguyên SL/TP do code tính."
            )
        await context.bot.send_message(chat_id=chat_id, text=message)
        decision_log = (
            review.decision if review is not None else "unavailable"
        ).encode("ascii", "backslashreplace").decode("ascii")
        logger.info(
            "Auto alert AI verification sent: %s model=%s decision=%s",
            side,
            used_label or "unavailable",
            decision_log,
        )
    except Exception as exc:
        logger.exception("Failed to send auto alert AI verification")
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"⚠️ AI XÁC THỰC {side} TẠM LỖI\n"
                f"• {type(exc).__name__}. Không coi đây là AI đồng thuận.\n"
                f"• Bot vẫn canh giá {active_poll_seconds} giây; giữ đúng kế hoạch SL/TP trong cảnh báo đầu tiên."
            ),
        )


async def auto_entry_alert_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    config = context.bot_data["config"]
    settings = config.get("auto_alerts", {})
    enabled = context.bot_data.get(
        "auto_alert_enabled_override",
        settings.get("enabled", True),
    )
    if not enabled:
        return

    lock = context.bot_data.setdefault("auto_alert_lock", asyncio.Lock())
    if lock.locked():
        logger.warning("Auto alert check skipped because previous check is still running")
        return

    async with lock:
        checked_at = datetime.now(timezone.utc)
        context.bot_data["auto_alert_last_check"] = checked_at
        try:
            opportunity = await asyncio.to_thread(
                compute_peak_opportunity,
                config,
                context.bot_data["provider"],
                context.bot_data["price_stream"],
                checked_at,
            )
            context.bot_data["auto_alert_last_error"] = None
            context.bot_data["auto_alert_last_gate"] = opportunity.gate.allowed_decision
            context.bot_data["auto_alert_last_reason"] = opportunity.gate.reason
            context.bot_data["auto_alert_last_execution_reason"] = opportunity.execution_reason
            active_wave = _get_active_wave(context, settings)
            if active_wave is not None:
                end_event, end_detail = _full_scan_wave_end_event(
                    active_wave,
                    opportunity,
                    settings,
                )
                if end_event is not None:
                    end_price = _wave_price(
                        active_wave,
                        opportunity.realtime_quote,
                        entering=False,
                    )
                    await context.bot.send_message(
                        chat_id=context.bot_data["authorized_chat_id"],
                        text=_format_active_wave_alert(
                            end_event,
                            active_wave,
                            end_price,
                            end_detail,
                            int(settings.get("active_poll_seconds", 10)),
                        ),
                    )
                    _record_diagnostic_event(
                        context,
                        f"active_wave_{end_event.lower()}",
                        {
                            "active_wave": active_wave,
                            "price": end_price,
                            "detail": end_detail,
                        },
                        checked_at,
                    )
                    _set_active_wave(context, settings, None)
                    active_wave = None
            phase, executable_price = _auto_alert_phase(opportunity, settings)
            _record_auto_analysis(
                context,
                opportunity,
                phase,
                executable_price,
            )
            if phase is None:
                logger.debug(
                    "Auto alert check: gate=%s reason=%s",
                    opportunity.gate.allowed_decision.encode(
                        "ascii", "backslashreplace"
                    ).decode("ascii"),
                    opportunity.gate.reason.encode(
                        "ascii", "backslashreplace"
                    ).decode("ascii"),
                )
                return

            plan = opportunity.execution_plan
            if plan is None:
                return
            fingerprint = _auto_alert_fingerprint(plan, phase)
            state_path = settings.get("state_path", "logs/auto_alert_state.json")
            state = _load_auto_alert_state(state_path)
            sent = state.get("sent", {}) if isinstance(state.get("sent", {}), dict) else {}
            cooldown = timedelta(
                minutes=max(1, int(settings.get("same_setup_cooldown_minutes", 240)))
            )
            last_sent_raw = sent.get(fingerprint)
            if last_sent_raw:
                try:
                    last_sent = datetime.fromisoformat(last_sent_raw)
                    if last_sent.tzinfo is None:
                        last_sent = last_sent.replace(tzinfo=timezone.utc)
                    if checked_at - last_sent < cooldown:
                        _record_auto_analysis(
                            context,
                            opportunity,
                            phase,
                            executable_price,
                            event="cooldown_suppressed",
                        )
                        return
                except ValueError:
                    pass

            chat_id = context.bot_data["authorized_chat_id"]
            await context.bot.send_message(
                chat_id=chat_id,
                text=format_auto_entry_alert(
                    opportunity,
                    phase,
                    executable_price,
                ),
            )
            _record_auto_analysis(
                context,
                opportunity,
                phase,
                executable_price,
                event="alert_sent",
                force=True,
            )
            if settings.get("send_hourly_chart", True):
                try:
                    chart = await asyncio.to_thread(
                        render_peak_confirmation_chart,
                        opportunity.frames,
                        opportunity.peak_map,
                        opportunity.liquidity,
                    )
                    await context.bot.send_photo(
                        chat_id=chat_id,
                        photo=chart,
                        caption="AUTO XAUUSDT · 15m Entry / 1H xác nhận / D1 xu hướng.",
                    )
                except Exception:
                    logger.exception("Auto alert text sent but 1H chart failed")

            cutoff = checked_at - timedelta(hours=48)
            retained = {}
            for key, value in sent.items():
                try:
                    value_time = datetime.fromisoformat(value)
                    if value_time.tzinfo is None:
                        value_time = value_time.replace(tzinfo=timezone.utc)
                    if value_time >= cutoff:
                        retained[key] = value
                except (TypeError, ValueError):
                    continue
            retained[fingerprint] = checked_at.isoformat()
            state["sent"] = retained
            state["last_alert"] = {
                "at": checked_at.isoformat(),
                "side": plan.side,
                "phase": phase,
                "entry": [plan.entry_lower, plan.entry_upper],
            }
            request_ai_review = False
            if active_wave is None:
                active_wave = _build_active_wave(
                    opportunity,
                    phase,
                    checked_at,
                    settings,
                )
                context.bot_data["active_wave"] = active_wave
            if (
                settings.get("ai_review_after_alert", True)
                and not active_wave.get("ai_review_requested_at")
            ):
                active_wave["ai_review_requested_at"] = checked_at.isoformat()
                request_ai_review = True
            state["active_wave"] = active_wave
            _save_auto_alert_state(state_path, state)
            context.bot_data["auto_alert_last_sent"] = checked_at
            logger.info("Auto entry alert sent: %s %s", plan.side, phase)
            if request_ai_review:
                context.job_queue.run_once(
                    auto_alert_ai_review_job,
                    when=0,
                    data={
                        "chat_id": chat_id,
                        "side": plan.side,
                        "gate": opportunity.gate,
                        "snapshot": _build_auto_peak_ai_snapshot(opportunity),
                    },
                    name=(
                        "xauusdt-auto-ai-review-"
                        + checked_at.strftime("%Y%m%d%H%M%S%f")
                    ),
                )
        except Exception as exc:
            context.bot_data["auto_alert_last_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            _record_diagnostic_event(
                context,
                "analysis_error",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                checked_at,
            )
            logger.exception("Auto entry alert check failed")


async def active_wave_monitor_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Watch an alerted setup on the configured fast loop without any LLM call."""
    config = context.bot_data["config"]
    settings = config.get("auto_alerts", {})
    enabled = context.bot_data.get(
        "auto_alert_enabled_override",
        settings.get("enabled", True),
    )
    if not enabled:
        return
    retry_after = context.bot_data.get("active_wave_retry_after")
    if retry_after is not None and datetime.now(timezone.utc) < retry_after:
        return
    active_wave = _get_active_wave(context, settings)
    if active_wave is None:
        return

    lock = context.bot_data.setdefault("auto_alert_lock", asyncio.Lock())
    if lock.locked():
        return
    async with lock:
        checked_at = datetime.now(timezone.utc)
        try:
            quote = await asyncio.to_thread(
                get_fast_quote,
                context.bot_data["provider"],
                context.bot_data["price_stream"],
            )
            context.bot_data["active_wave_last_error"] = None
            context.bot_data["active_wave_retry_after"] = None
            active_wave = _get_active_wave(context, settings)
            if active_wave is None:
                return
            price = _wave_price(
                active_wave,
                quote,
                entering=not bool(active_wave.get("entered")),
            )
            active_wave["last_check_at"] = checked_at.isoformat()
            active_wave["last_price"] = price
            context.bot_data["active_wave_last_check"] = checked_at
            event = _fast_wave_event(active_wave, price, settings)
            detail = ""

            if event is None and active_wave.get("entered"):
                (
                    pressure_score,
                    monotonic_opposite,
                    fetched,
                    last_micro_close,
                    last_micro_closed_at,
                ) = await asyncio.to_thread(
                    _refresh_micro_pressure,
                    context.bot_data["provider"],
                    active_wave,
                    checked_at,
                    settings,
                )
                threshold = max(
                    0.1,
                    float(settings.get("micro_weakness_threshold", 0.35)),
                )
                adverse_r = max(
                    0.1,
                    float(settings.get("adverse_warning_r", 0.35)),
                )
                progress_r = _wave_progress_r(active_wave, price)
                pressure_opposite = (
                    pressure_score is not None
                    and (
                        pressure_score <= -threshold
                        if active_wave["side"] == "LONG"
                        else pressure_score >= threshold
                    )
                )
                retest_failed, retest_detail = _micro_retest_failure(
                    active_wave,
                    pressure_score,
                    last_micro_close,
                    last_micro_closed_at,
                    settings,
                )
                if fetched and retest_failed:
                    event = "RETEST_FAILED"
                    detail = retest_detail
                elif (
                    not active_wave.get("weakness_notified")
                    and progress_r <= -adverse_r
                    and pressure_opposite
                    and monotonic_opposite
                ):
                    event = "WEAK"
                    detail = (
                        f"Áp lực nến 1m đang ngược kèo "
                        f"({float(pressure_score):+.2f})."
                    )
                if fetched:
                    _set_active_wave(context, settings, active_wave)

            if event is None:
                context.bot_data["active_wave"] = active_wave
                return

            if event == "ENTRY":
                active_wave["entered"] = True
                active_wave["phase"] = "ACTIVE"
                active_wave["entered_at"] = checked_at.isoformat()
                holding = min(
                    max(1, int(active_wave.get("max_holding_minutes", 240))),
                    max(1, int(settings.get("active_wave_max_minutes", 480))),
                )
                active_wave["expires_at"] = (
                    checked_at + timedelta(minutes=holding)
                ).isoformat()
            elif event == "TP1":
                active_wave["tp1_notified"] = True
            elif event == "WEAK":
                active_wave["weakness_notified"] = True

            await context.bot.send_message(
                chat_id=context.bot_data["authorized_chat_id"],
                text=_format_active_wave_alert(
                    event,
                    active_wave,
                    price,
                    detail,
                    int(settings.get("active_poll_seconds", 10)),
                ),
            )
            _record_diagnostic_event(
                context,
                f"active_wave_{event.lower()}",
                {
                    "active_wave": active_wave,
                    "price": price,
                    "detail": detail,
                },
                checked_at,
            )
            terminal_events = {
                "TP2",
                "BREAKEVEN",
                "STOP",
                "RETEST_FAILED",
                "INVALIDATED",
                "RUNAWAY",
                "EXPIRED",
                "TIME_STOP",
            }
            _set_active_wave(
                context,
                settings,
                None if event in terminal_events else active_wave,
            )
            logger.info("Active wave event sent: %s %s", active_wave["side"], event)
        except Exception as exc:
            context.bot_data["active_wave_last_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            context.bot_data["active_wave_retry_after"] = (
                datetime.now(timezone.utc)
                + timedelta(
                    seconds=max(15, int(settings.get("error_backoff_seconds", 60)))
                )
            )
            _record_diagnostic_event(
                context,
                "active_wave_error",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                datetime.now(timezone.utc),
            )
            logger.exception("Active wave fast check failed")


async def manual_position_monitor_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Poll a user-declared position until /dong; never assumes exchange access."""
    config = context.bot_data["config"]
    settings = _manual_position_settings(config)
    if not settings.get("enabled", True):
        return
    retry_after = context.bot_data.get("manual_position_retry_after")
    if retry_after is not None and datetime.now(timezone.utc) < retry_after:
        return
    position = _get_manual_position(context, settings)
    if position is None:
        return
    lock = context.bot_data.setdefault("manual_position_lock", asyncio.Lock())
    if lock.locked():
        return
    async with lock:
        checked_at = datetime.now(timezone.utc)
        try:
            quote = await asyncio.to_thread(
                get_fast_quote,
                context.bot_data["provider"],
                context.bot_data["price_stream"],
            )
            context.bot_data["manual_position_last_error"] = None
            context.bot_data["manual_position_retry_after"] = None
            position = _get_manual_position(context, settings)
            if position is None:
                return
            price = _manual_exit_price(position, quote)
            metrics = manual_position_metrics(position, price)
            event = update_manual_position_event(
                position,
                metrics,
                checked_at,
                settings,
            )
            position["last_check_at"] = checked_at.isoformat()
            position["last_price"] = price
            position["last_metrics"] = metrics
            context.bot_data["manual_position_last_check"] = checked_at
            if not manual_alert_is_due(position, event, checked_at, settings):
                context.bot_data["manual_position"] = position
                return
            await context.bot.send_message(
                chat_id=context.bot_data["authorized_chat_id"],
                text=format_manual_position_alert(
                    position,
                    metrics,
                    event,
                    checked_at,
                    settings,
                ),
            )
            position["last_alert_event"] = event
            position["last_alert_at"] = checked_at.isoformat()
            _set_manual_position(context, settings, position)
            _record_diagnostic_event(
                context,
                f"manual_position_{event.lower()}",
                {"position": position, "metrics": metrics},
                checked_at,
            )
            logger.info(
                "Manual position event sent: %s %s",
                position["side"],
                event,
            )
        except Exception as exc:
            context.bot_data["manual_position_last_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            context.bot_data["manual_position_retry_after"] = (
                datetime.now(timezone.utc)
                + timedelta(
                    seconds=max(
                        10,
                        int(settings.get("error_backoff_seconds", 30)),
                    )
                )
            )
            last_notice = context.bot_data.get("manual_position_error_notice_at")
            notice_interval = max(
                30,
                int(settings.get("error_notice_seconds", 300)),
            )
            if (
                last_notice is None
                or (datetime.now(timezone.utc) - last_notice).total_seconds()
                >= notice_interval
            ):
                try:
                    await context.bot.send_message(
                        chat_id=context.bot_data["authorized_chat_id"],
                        text=(
                            "⚠️ MONITOR VỊ THẾ TẠM KHÔNG LẤY ĐƯỢC GIÁ\n"
                            f"• Lỗi {type(exc).__name__}; vị thế vẫn được lưu và bot sẽ tự thử lại.\n"
                            "• Hãy kiểm tra trực tiếp Binance; nếu đã đóng lệnh thì gửi /dong."
                        ),
                    )
                    context.bot_data["manual_position_error_notice_at"] = datetime.now(
                        timezone.utc
                    )
                except Exception:
                    logger.exception("Could not send manual-position data error notice")
            logger.exception("Manual position monitor check failed")


async def manual_position_startup_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = _manual_position_settings(context.bot_data["config"])
    position = _get_manual_position(context, settings)
    if position is None:
        return
    await context.bot.send_message(
        chat_id=context.bot_data["authorized_chat_id"],
        text=(
            f"♻️ ĐÃ KHÔI PHỤC THEO DÕI {position['side']} sau khi bot khởi động lại.\n"
            f"• Entry {float(position['entry_price']):.2f} · ký quỹ {float(position['margin_usdt']):.2f} USDT · {int(position['leverage'])}x.\n"
            f"• Bot tiếp tục check mỗi {int(settings.get('poll_seconds', 10))} giây cho đến /dong."
        ),
    )


async def auto_monitor_startup_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.bot_data["config"].get("auto_alerts", {})
    if not settings.get("enabled", True) or not settings.get("notify_on_start", True):
        return
    await context.bot.send_message(
        chat_id=context.bot_data["authorized_chat_id"],
        text=(
            "✅ CANH TỰ ĐỘNG XAUUSDT ĐÃ BẬT\n"
            f"• Kiểm tra mỗi {int(settings.get('poll_seconds', 30))} giây.\n"
            f"• Sau khi báo LONG/SHORT: canh giá mỗi {int(settings.get('active_poll_seconds', 10))} giây; AI không bị gọi nền.\n"
            "• Chỉ báo khi đủ 15m + 1H + 4H, D1 không đối nghịch, thanh khoản đạt và giá sát/vào Entry.\n"
            "• Từ 05:00 thứ Bảy đến hết Chủ Nhật không phát cảnh báo vào lệnh.\n"
            "• Dùng /canh để xem trạng thái; /canhtat hoặc /canhbat để điều khiển."
        ),
    )


async def handle_auto_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.bot_data["config"].get("auto_alerts", {})
    enabled = context.bot_data.get(
        "auto_alert_enabled_override",
        settings.get("enabled", True),
    )
    last_check = context.bot_data.get("auto_alert_last_check")
    last_sent = context.bot_data.get("auto_alert_last_sent")
    last_error = context.bot_data.get("auto_alert_last_error")
    last_gate = context.bot_data.get("auto_alert_last_gate", "chưa kiểm tra")
    last_reason = context.bot_data.get("auto_alert_last_reason", "chưa có dữ liệu")
    last_execution_reason = context.bot_data.get(
        "auto_alert_last_execution_reason",
        "chưa có dữ liệu",
    )
    diagnostic_path = context.bot_data.get(
        "last_analysis_diagnostic_path",
        "chưa ghi",
    )
    active_wave = _get_active_wave(context, settings)
    if active_wave is not None:
        wave_last_check = _parse_utc(active_wave.get("last_check_at"))
        wave_status = (
            f"{active_wave['side']} · "
            f"{'đang trong lệnh' if active_wave.get('entered') else 'chờ Entry'} · "
            f"giá gần nhất {float(active_wave.get('last_price', 0)):.2f} · "
            f"check {wave_last_check.astimezone().strftime('%H:%M:%S') if wave_last_check else 'chưa có'}"
        )
    else:
        wave_status = "không có kèo đang canh nhanh"
    await update.message.reply_text(
        "\n".join(
            [
                f"🔔 Canh tự động: {'ĐANG BẬT' if enabled else 'ĐANG TẮT'}",
                f"• Chu kỳ: {int(settings.get('poll_seconds', 30))} giây.",
                f"• Canh nhanh: {int(settings.get('active_poll_seconds', 10))} giây — {wave_status}.",
                f"• Lần kiểm tra: {last_check.astimezone().strftime('%d/%m %H:%M:%S') if last_check else 'chưa chạy'}.",
                f"• Gate gần nhất: {last_gate} — {last_reason}",
                f"• Kế hoạch lệnh: {last_execution_reason}",
                f"• Diagnostic: {diagnostic_path}",
                f"• Cảnh báo gần nhất: {last_sent.astimezone().strftime('%d/%m %H:%M:%S') if last_sent else 'chưa có'}.",
                f"• Lỗi gần nhất: {last_error or 'không có'}.",
            ]
        )
    )


async def handle_auto_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot_data["auto_alert_enabled_override"] = True
    await update.message.reply_text("✅ Đã bật canh tự động; bot sẽ kiểm tra ở chu kỳ kế tiếp.")


async def handle_auto_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.bot_data["auto_alert_enabled_override"] = False
    await update.message.reply_text("⏸ Đã tắt canh tự động. /gia và /dinh vẫn dùng bình thường.")


async def handle_manual_position(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    text = update.effective_message.text if update.effective_message else ""
    parsed = parse_manual_position_command(text)
    usage = (
        "Cú pháp: /long:100_5x hoặc /long 100 5x; "
        "/short:100_5x (cũng chấp nhận /sort:100_5x)."
    )
    if parsed is None:
        await update.effective_message.reply_text(usage)
        return
    side, margin_usdt, leverage = parsed
    config = context.bot_data["config"]
    settings = _manual_position_settings(config)
    if not settings.get("enabled", True):
        await update.effective_message.reply_text("Theo dõi vị thế thủ công đang tắt trong config.")
        return
    minimum_margin = max(0.01, float(settings.get("minimum_margin_usdt", 1.0)))
    maximum_margin = max(minimum_margin, float(settings.get("maximum_margin_usdt", 1_000_000)))
    maximum_leverage = max(1, int(settings.get("maximum_leverage", 125)))
    if not minimum_margin <= margin_usdt <= maximum_margin:
        await update.effective_message.reply_text(
            f"Số tiền phải từ {minimum_margin:.2f} đến {maximum_margin:.2f} USDT. {usage}"
        )
        return
    if leverage < 1 or leverage > maximum_leverage:
        await update.effective_message.reply_text(
            f"Đòn bẩy phải từ 1x đến {maximum_leverage}x. {usage}"
        )
        return

    lock = context.bot_data.setdefault("manual_position_lock", asyncio.Lock())
    async with lock:
        existing = _get_manual_position(context, settings)
        if existing is not None:
            await update.effective_message.reply_text(
                f"Đang theo dõi {existing['side']} từ {float(existing['entry_price']):.2f}, "
                f"ký quỹ {float(existing['margin_usdt']):.2f} USDT · {int(existing['leverage'])}x. "
                "Gửi /dong sau khi đóng lệnh thật rồi mới khai báo vị thế mới."
            )
            return
        try:
            _quote, entry_price, atr, checked_at = await asyncio.to_thread(
                _capture_manual_position_market,
                config,
                context.bot_data["provider"],
                context.bot_data["price_stream"],
                side,
            )
            position = build_manual_position_state(
                side,
                margin_usdt,
                leverage,
                entry_price,
                atr,
                checked_at,
                settings,
            )
            # The opening reply is the first status message; avoid duplicating it
            # ten seconds later while still polling at the requested cadence.
            position["last_alert_event"] = "STATUS"
            position["last_alert_at"] = checked_at.isoformat()
            _set_manual_position(context, settings, position)
            _record_diagnostic_event(
                context,
                "manual_position_opened",
                {"position": position},
                checked_at,
            )
        except ValueError as exc:
            await update.effective_message.reply_text(f"Không thể ghi nhận vị thế: {exc}")
            return
        except Exception as exc:
            logger.exception("Failed to start manual position monitor")
            await update.effective_message.reply_text(
                f"Không lấy được giá/nến để bắt đầu theo dõi ({type(exc).__name__}); thử lại sau."
            )
            return

    leverage_warning = (
        "\n⚠️ Đòn bẩy cao: biến động nhỏ có thể làm ROE và rủi ro thanh lý tăng rất nhanh."
        if leverage >= int(settings.get("high_leverage_warning_x", 10))
        else ""
    )
    await update.effective_message.reply_text(
        "\n".join(
            [
                f"✅ ĐÃ GHI NHẬN {side} — THEO DÕI THAM CHIẾU",
                f"• Ký quỹ {margin_usdt:.2f} USDT · {leverage}x · notional {float(position['notional_usdt']):.2f} USDT · {float(position['quantity_xau']):.3f} XAU.",
                f"• Entry tham chiếu {entry_price:.2f} · ATR15m {atr:.2f}. Đây là quote khi bot nhận lệnh, không xác nhận fill Binance thật.",
                f"• SL {float(position['stop_loss']):.2f} · TP1 {float(position['take_profit_1']):.2f} · TP2 {float(position['take_profit_2']):.2f}.",
                f"• Nếu chạm SL: lỗ ước tính sau phí/trượt khoảng {float(position['projected_stop_loss_usdt']):.2f} USDT ({float(position['projected_stop_roe_pct']):.2f}% ký quỹ).",
                f"• Bot kiểm tra mỗi {int(settings.get('poll_seconds', 10))} giây. Khi cần đóng, cảnh báo lặp đến khi bạn gửi /dong.",
                "• /vithe để xem ngay; /dong chỉ dừng theo dõi, KHÔNG đóng lệnh trên Binance."
                + leverage_warning,
            ]
        )
    )


async def handle_manual_position_status(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    config = context.bot_data["config"]
    settings = _manual_position_settings(config)
    lock = context.bot_data.setdefault("manual_position_lock", asyncio.Lock())
    async with lock:
        position = _get_manual_position(context, settings)
        if position is None:
            await update.effective_message.reply_text(
                "Không có vị thế thủ công đang theo dõi. Dùng /long:100_5x hoặc /short:100_5x."
            )
            return
        checked_at = datetime.now(timezone.utc)
        try:
            quote = await asyncio.to_thread(
                get_fast_quote,
                context.bot_data["provider"],
                context.bot_data["price_stream"],
            )
            price = _manual_exit_price(position, quote)
        except Exception:
            price = float(position.get("last_price", position["entry_price"]))
        metrics = manual_position_metrics(position, price)
        event = update_manual_position_event(position, metrics, checked_at, settings)
        position["last_check_at"] = checked_at.isoformat()
        position["last_price"] = price
        _set_manual_position(context, settings, position)
    await update.effective_message.reply_text(
        format_manual_position_alert(position, metrics, event, checked_at, settings)
    )


async def handle_manual_position_close(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    config = context.bot_data["config"]
    settings = _manual_position_settings(config)
    lock = context.bot_data.setdefault("manual_position_lock", asyncio.Lock())
    async with lock:
        position = _get_manual_position(context, settings)
        if position is None:
            await update.effective_message.reply_text("Không có vị thế thủ công nào đang theo dõi.")
            return
        # Stop first so a slow/failed quote request can never leave reminders active.
        _set_manual_position(context, settings, None)
    checked_at = datetime.now(timezone.utc)
    try:
        quote = await asyncio.to_thread(
            get_fast_quote,
            context.bot_data["provider"],
            context.bot_data["price_stream"],
        )
        price = _manual_exit_price(position, quote)
    except Exception:
        price = float(position.get("last_price", position["entry_price"]))
    metrics = manual_position_metrics(position, price)
    _record_diagnostic_event(
        context,
        "manual_position_monitor_stopped",
        {"position": position, "metrics": metrics},
        checked_at,
    )
    await update.effective_message.reply_text(
        "\n".join(
            [
                f"⏹ ĐÃ DỪNG THEO DÕI {position['side']}",
                f"• Giá cuối tham chiếu {price:.2f} · PnL gộp {float(metrics['gross_pnl_usdt']):+.2f} USDT · ước tính sau phí/trượt {float(metrics['estimated_net_pnl_usdt']):+.2f} USDT.",
                "• /dong chỉ dừng cảnh báo của bot; hãy tự xác nhận vị thế và các lệnh SL/TP trên Binance đã đóng/hủy đúng.",
            ]
        )
    )


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Bot phân tích Binance Futures XAUUSDT.\n"
        "• /gia hoặc /signal: kịch bản giao dịch đã xác nhận.\n"
        "• /dinh: bản đồ đỉnh cũ/mới, cản trên và hỗ trợ retest.\n"
        "• /long:100_5x hoặc /short:100_5x: theo dõi vị thế thủ công mỗi 10 giây.\n"
        "• /vithe: xem PnL; /dong: dừng theo dõi thủ công.\n"
        "• /canh: trạng thái canh tự động; /canhbat hoặc /canhtat để điều khiển."
    )


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    authorized_chat_id = int(os.environ["TELEGRAM_CHAT_ID"])
    config = load_config()
    market_data_config = config.get("market_data", {})
    provider = BinanceFuturesProvider(
        symbol=config.get("symbol", "XAUUSDT"),
        base_url=market_data_config.get(
            "base_url",
            "https://fapi.binance.com",
        ),
    )
    price_stream = BinanceFuturesPriceStream(
        symbol=provider.symbol,
        websocket_base_url=market_data_config.get(
            "websocket_base_url",
            "wss://fstream.binance.com",
        ),
    )
    price_stream.start()

    app = Application.builder().token(token).build()
    app.bot_data["config"] = config
    app.bot_data["provider"] = provider
    app.bot_data["price_stream"] = price_stream
    app.bot_data["authorized_chat_id"] = authorized_chat_id

    chat_filter = filters.Chat(chat_id=authorized_chat_id)
    app.add_handler(CommandHandler("start", handle_start, filters=chat_filter))
    app.add_handler(CommandHandler(["signal", "gia"], handle_signal, filters=chat_filter))
    app.add_handler(CommandHandler("dinh", handle_peaks, filters=chat_filter))
    app.add_handler(CommandHandler("canh", handle_auto_status, filters=chat_filter))
    app.add_handler(CommandHandler("canhbat", handle_auto_on, filters=chat_filter))
    app.add_handler(CommandHandler("canhtat", handle_auto_off, filters=chat_filter))
    app.add_handler(
        CommandHandler(
            ["long", "short", "sort"],
            handle_manual_position,
            filters=chat_filter,
        )
    )
    app.add_handler(CommandHandler("vithe", handle_manual_position_status, filters=chat_filter))
    app.add_handler(CommandHandler("dong", handle_manual_position_close, filters=chat_filter))
    app.add_handler(
        MessageHandler(
            chat_filter & filters.Regex(MANUAL_POSITION_TEXT_PATTERN),
            handle_manual_position,
        )
    )

    auto_settings = config.get("auto_alerts", {})
    if auto_settings.get("enabled", True):
        poll_seconds = max(15, int(auto_settings.get("poll_seconds", 30)))
        first_check_seconds = max(
            1,
            int(auto_settings.get("first_check_seconds", 10)),
        )
        app.job_queue.run_repeating(
            auto_entry_alert_job,
            interval=poll_seconds,
            first=first_check_seconds,
            name="xauusdt-auto-entry-alert",
        )
        active_poll_seconds = max(
            5,
            int(auto_settings.get("active_poll_seconds", 10)),
        )
        app.job_queue.run_repeating(
            active_wave_monitor_job,
            interval=active_poll_seconds,
            first=max(2, min(active_poll_seconds, first_check_seconds + 2)),
            name="xauusdt-active-wave-monitor",
        )
        if auto_settings.get("notify_on_start", True):
            app.job_queue.run_once(
                auto_monitor_startup_job,
                when=3,
                name="xauusdt-auto-monitor-started",
            )

    manual_settings = _manual_position_settings(config)
    if manual_settings.get("enabled", True):
        manual_poll_seconds = max(
            5,
            int(manual_settings.get("poll_seconds", 10)),
        )
        app.job_queue.run_repeating(
            manual_position_monitor_job,
            interval=manual_poll_seconds,
            first=max(2, min(manual_poll_seconds, 5)),
            name="xauusdt-manual-position-monitor",
        )
        app.job_queue.run_once(
            manual_position_startup_job,
            when=4,
            name="xauusdt-manual-position-restored",
        )

    logger.info(
        "Telegram query bot started for chat_id=%s; auto alerts=%s interval=%ss active=%ss manual=%ss",
        authorized_chat_id,
        auto_settings.get("enabled", True),
        int(auto_settings.get("poll_seconds", 30)),
        int(auto_settings.get("active_poll_seconds", 10)),
        int(manual_settings.get("poll_seconds", 10)),
    )
    try:
        app.run_polling()
    finally:
        price_stream.stop()


if __name__ == "__main__":
    main()
