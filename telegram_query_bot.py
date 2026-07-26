import asyncio
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import yaml
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, filters

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
    PeakExecutionPlan,
    PeakLiquidityAssessment,
    PeakTradeGate,
    assess_peak_liquidity,
    assess_peak_trade_gate,
    build_peak_execution_plan,
    build_peak_map,
    format_peak_map,
    render_peak_hourly_chart,
)

load_dotenv()

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/telegram_query_bot.log"), logging.StreamHandler()],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("gold-query-bot")

SIGNAL_LABEL = {"BUY": "nghieng TANG", "SELL": "nghieng GIAM", "HOLD": "trung lap, chua ro huong"}


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_current_quote(provider, price_stream) -> RealtimeQuote:
    stream_quote = price_stream.latest()
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
    buffer = 1 + float(settings.get("slippage_fee_buffer_pct", 10)) / 100
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
    quantity = risk_budget / (stop_distance * buffer)

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
    actual_risk = quantity * stop_distance * buffer
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
) -> str:
    resistance = gate.resistance
    ratio_text = (
        f"{liquidity.volume_ratio:.0%} median ngày thường"
        if liquidity.volume_ratio is not None
        else "chưa đủ mẫu so sánh"
    )
    lines = [
        "🎯 HƯỚNG DẪN THỰC THI",
        f"• 1H: {hourly_structure.pattern} · bias {hourly_score * 100:+.0f}% · "
        f"S {hourly_structure.support:.2f} / R {hourly_structure.resistance:.2f} · "
        f"xác nhận {'ĐẠT' if gate.hourly_confirmed else 'CHƯA ĐẠT'}.",
        f"• Thanh khoản: {liquidity.status} · volume {ratio_text}. {liquidity.reason}",
    ]
    if resistance is None:
        return "\n".join(lines + ["• KHÔNG VÀO: chưa có vùng cản đủ tin cậy để lập kế hoạch."])

    if execution_plan is None:
        lines += [
            f"• Hiện tại: KHÔNG VÀO — {execution_reason}",
            f"• SHORT: chờ giá hồi vào {resistance.lower:.2f}–{resistance.upper:.2f}; "
            f"15m từ chối và 1H đóng dưới {resistance.lower:.2f}, rồi retest không vượt lại vùng.",
            f"• LONG: chờ 15m đóng trên {resistance.upper:.2f}; nến sau retest "
            f"{resistance.lower:.2f}–{resistance.upper:.2f}, đồng thời 1H đóng trên {resistance.upper:.2f}.",
            "• Lúc này chưa đặt Limit/SL/TP. Đủ điều kiện thì gọi /dinh lại; râu nến không tính xác nhận.",
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
    lines += [
        f"• {plan.side}: chờ giá quay lại {plan.entry_lower:.2f}–{plan.entry_upper:.2f} "
        f"và nến 15m retest {retest_action}; 1H đã đóng xác nhận, không đuổi giá.",
        f"• Entry tham chiếu {plan.entry_reference:.2f} · SL {plan.stop_loss:.2f} "
        "(Stop-Market, ưu tiên trigger Mark Price).",
        f"• TP1 {plan.take_profit_1:.2f} ({plan.reward_risk_1:.1f}R) · "
        f"TP2 {plan.take_profit_2:.2f} ({plan.reward_risk_2:.1f}R, {plan.structural_target}).",
    ]
    if sized_plan is not None:
        lines += [
            f"• Vốn: {sized_plan.quantity_xau:.3f} XAU · {sized_plan.leverage}x isolated · "
            f"rủi ro tối đa {sized_plan.risk_usdt:.2f} USDT "
            f"({sized_plan.actual_risk_pct:.2f}% trên số dư cấu hình "
            f"{sized_plan.account_balance_usdt:.2f} USDT).",
            f"• Quản lệnh: chốt 50% ở TP1, phần còn lại dời SL về Entry; "
            f"time-stop ~{format_duration(sized_plan.max_holding_minutes)}, không giữ qua ngày.",
            "",
            "CÁCH ĐẶT TRÊN BINANCE",
            f"1) Chọn Isolated {sized_plan.leverage}x → Limit {entry_button} "
            f"{sized_plan.quantity_xau:.3f} XAU trong vùng Entry; hết vùng thì hủy, không Market đuổi.",
            f"2) Sau khi Entry khớp → đặt Stop-Market {close_button} toàn bộ vị thế tại "
            f"{plan.stop_loss:.2f}; trigger Mark Price và chọn Close Position/Reduce-Only nếu giao diện có.",
            f"3) Đặt TP Limit {close_button}: 50% tại {plan.take_profit_1:.2f}; "
            f"phần còn lại tại {plan.take_profit_2:.2f}, đều chỉ giảm vị thế.",
            "4) TP1 khớp → dời SL phần còn lại về Entry; vị thế đã đóng thì hủy mọi lệnh SL/TP còn treo.",
        ]
    else:
        lines.append("• Không tính được khối lượng an toàn với số dư/cấu hình hiện tại; không vào lệnh.")
    lines.append("• Hủy kế hoạch nếu nến 15m đóng phá SL trước khi khớp Entry.")
    return "\n".join(lines)


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

    lines = [
        f"📊 *XAUUSDT PERP {bias.price:.2f}* · {feed_status} · {realtime_quote.source}",
        f"Nguồn lúc {quote_time.astimezone().strftime('%d/%m %H:%M:%S')}",
        *fair_value_lines,
        f"🧭 *QUYẾT ĐỊNH: {forecast}* — {retest.decision_reason}",
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
        timeframes = settings.get(
            "timeframes",
            ["15min", "1h", "4h", "1day"],
        )
        outputsize = int(settings.get("outputsize", 200))
        analysis_now = datetime.now(timezone.utc)
        frames = {
            timeframe: select_closed_candles(
                provider.get_historical(
                    interval=timeframe,
                    outputsize=outputsize,
                ),
                timeframe,
                analysis_now,
            )
            for timeframe in timeframes
        }
        realtime_quote = get_current_quote(provider, price_stream)
        peak_map = build_peak_map(
            frames=frames,
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
        hourly_structure = find_chart_structure(frames["1h"])
        reply = format_peak_map(peak_map, settings)
        logger.info(
            "Peak map requested by chat %s: peaks=%s resistance=%s support=%s",
            update.effective_chat.id,
            peak_map.scanned_peak_count,
            len(peak_map.resistance_zones),
            len(peak_map.converted_support_zones),
        )
        try:
            hourly_chart = render_peak_hourly_chart(
                frames["1h"],
                peak_map,
                liquidity,
            )
            await update.message.reply_photo(
                photo=hourly_chart,
                caption=(
                    "XAUUSDT 1H · nến đã đóng · EMA9/EMA21 · "
                    "kháng cự/hỗ trợ và volume thật."
                ),
            )
        except Exception:
            logger.exception("Could not render/send peak 1H chart")
        await update.message.reply_text(
            reply,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )
        analysis_frames = {
            timeframe: frame
            for timeframe, frame in frames.items()
            if timeframe in ("15min", "1h", "4h")
        }
        momentum_biases = {
            timeframe: compute_momentum_bias(
                frame,
                config["weights"],
            )
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
        )
        execution_settings = {
            **config.get("trading_plan", {}),
            **config.get("peak_execution", {}),
        }
        execution_plan, execution_reason = build_peak_execution_plan(
            peak_map,
            gate,
            frames["15min"],
            execution_settings,
        )
        sized_plan = None
        if execution_plan is not None:
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
                },
                actionable=True,
                side_override=execution_plan.side,
                entry_override=execution_plan.entry_reference,
                stop_override=execution_plan.stop_loss,
                take_profit_1_override=execution_plan.take_profit_1,
                take_profit_2_override=execution_plan.take_profit_2,
            )
        execution_guide = format_peak_execution_guide(
            gate,
            execution_plan,
            execution_reason,
            sized_plan,
            liquidity,
            hourly_structure,
            momentum_scores["1h"],
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


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Bot phân tích Binance Futures XAUUSDT.\n"
        "• /gia hoặc /signal: kịch bản giao dịch đã xác nhận.\n"
        "• /dinh: bản đồ đỉnh cũ/mới, cản trên và hỗ trợ retest."
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

    chat_filter = filters.Chat(chat_id=authorized_chat_id)
    app.add_handler(CommandHandler("start", handle_start, filters=chat_filter))
    app.add_handler(CommandHandler(["signal", "gia"], handle_signal, filters=chat_filter))
    app.add_handler(CommandHandler("dinh", handle_peaks, filters=chat_filter))

    logger.info(
        "Telegram query bot started for chat_id=%s, waiting for /signal, /gia or /dinh...",
        authorized_chat_id,
    )
    try:
        app.run_polling()
    finally:
        price_stream.stop()


if __name__ == "__main__":
    main()
