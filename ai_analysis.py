import base64
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from market_context import (
    ChartStructure,
    NewsFetchResult,
    OrderBookPressure,
    PricePressure,
    ResistanceZoneAnalysis,
    RetestAssessment,
    TimeframeConsensus,
)
from peak_analysis import (
    PeakExecutionPlan,
    PeakLiquidityAssessment,
    PeakMap,
    PeakTradeGate,
)


class AIMarketAnalysis(BaseModel):
    summary_vi: str = Field(description="Tóm tắt thị trường bằng tiếng Việt, tối đa 3 câu ngắn.")
    chart_structure_vi: str = Field(
        description="Giải thích đỉnh đáy, hỗ trợ, kháng cự và cấu trúc biểu đồ bằng tiếng Việt."
    )
    momentum_vi: str = Field(description="Giải thích động lượng và sự đồng thuận hoặc mâu thuẫn đa khung.")
    news_context_vi: str = Field(
        description="Tóm tắt tác động tiềm năng của các tiêu đề tin, không thêm sự kiện ngoài dữ liệu."
    )
    primary_scenario_vi: str = Field(
        description="Kịch bản chính có điều kiện, không ra lệnh chắc chắn và không tự tạo mức giá."
    )
    alternative_scenario_vi: str = Field(
        description="Kịch bản ngược lại và dấu hiệu khiến kịch bản chính thất bại."
    )
    risk_vi: str = Field(description="Rủi ro quan trọng nhất và lý do nên đứng ngoài nếu dữ liệu mâu thuẫn.")
    stance: Literal["ĐỨNG NGOÀI", "NGHIÊNG LONG", "NGHIÊNG SHORT"] = Field(
        description="Nhận định tham khảo; không được trái với deterministic_actionable."
    )
    data_consistency: int = Field(
        ge=0,
        le=100,
        description="Mức nhất quán giữa các nguồn dữ liệu, không phải xác suất thắng.",
    )


class AIPeakReview(BaseModel):
    decision: Literal["CHỜ", "CANH LONG", "CANH SHORT"]
    review_vi: str = Field(description="Review vùng đỉnh và bối cảnh đa khung, tối đa 180 ký tự.")
    confirmation_vi: str = Field(description="Điều kiện nến đóng cần chờ, tối đa 180 ký tự.")
    invalidation_vi: str = Field(description="Điều kiện vô hiệu dùng đúng vùng code cung cấp, tối đa 160 ký tự.")
    risk_vi: str = Field(description="Rủi ro quan trọng nhất, tối đa 160 ký tự.")
    data_consistency: int = Field(
        ge=0,
        le=100,
        description="Độ nhất quán dữ liệu, không phải xác suất thắng.",
    )


def ai_daily_budget_available(path: str, daily_limit: int) -> bool:
    if daily_limit <= 0:
        return False
    today = datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        count = int(payload.get("count", 0)) if payload.get("date") == today else 0
    except (OSError, ValueError, TypeError):
        count = 0
    return count < daily_limit


def record_ai_call(path: str) -> None:
    today = datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    usage_path = Path(path)
    try:
        payload = json.loads(usage_path.read_text(encoding="utf-8"))
        count = int(payload.get("count", 0)) if payload.get("date") == today else 0
    except (OSError, ValueError, TypeError):
        count = 0
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    usage_path.write_text(
        json.dumps({"date": today, "count": count + 1}),
        encoding="utf-8",
    )


def is_gemini_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "quota exceeded" in message or "too_many_requests" in message


def is_ai_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        is_gemini_rate_limit_error(error)
        or "rate limit" in message
        or "rate_limit" in message
    )


def _structure_payload(structure: ChartStructure) -> dict:
    return {
        "pattern": structure.pattern,
        "trend": structure.trend,
        "support": round(structure.support, 4),
        "resistance": round(structure.resistance, 4),
        "recent_swing_highs": [round(point.price, 4) for point in structure.swing_highs],
        "recent_swing_lows": [round(point.price, 4) for point in structure.swing_lows],
    }


def build_ai_snapshot(
    df: pd.DataFrame,
    current_price: float,
    quote_source: str,
    consensus: TimeframeConsensus,
    structures: dict[str, ChartStructure],
    pressure: PricePressure,
    order_book: OrderBookPressure | None,
    indicator_components: dict[str, float],
    atr: float,
    legacy_signal: str,
    news_result: NewsFetchResult,
    plan,
    resistance_test: ResistanceZoneAnalysis | None = None,
    frames: dict[str, pd.DataFrame] | None = None,
    retest: RetestAssessment | None = None,
    derivatives_metrics: dict | None = None,
) -> dict:
    def candle_payload(frame: pd.DataFrame) -> list[dict]:
        return [
            {
                "time": timestamp.isoformat(),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
            }
            for timestamp, row in frame.tail(8).iterrows()
        ]

    frames = frames or {"15min": df}

    plan_payload = None
    if plan is not None:
        plan_payload = {
            "side": plan.side,
            "entry": round(plan.entry, 4),
            "stop": round(plan.stop, 4),
            "take_profit_1": round(plan.take_profit_1, 4),
            "take_profit_2": round(plan.take_profit_2, 4),
            "risk_pct": round(plan.actual_risk_pct, 4),
            "leverage": plan.leverage,
        }

    return {
        "instrument": "Binance Futures XAUUSDT perpetual",
        "current_price": round(current_price, 4),
        "quote_source": quote_source,
        "binance_futures_metrics": derivatives_metrics,
        "deterministic_actionable": plan is not None,
        "deterministic_consensus_score": round(consensus.score, 4),
        "closed_candle_consensus_actionable": consensus.actionable,
        "timeframe_momentum_scores": {
            key: round(value, 4) for key, value in consensus.scores.items()
        },
        "structures": {
            timeframe: _structure_payload(structure)
            for timeframe, structure in structures.items()
        },
        "price_pressure_1m": {
            "score": round(pressure.score, 4),
            "up_bars": pressure.up_bars,
            "down_bars": pressure.down_bars,
            "taker_buy_notional": pressure.taker_buy_notional,
            "taker_sell_notional": pressure.taker_sell_notional,
            "uses_actual_trade_flow": pressure.uses_trade_flow,
            "note": "Closed Binance XAUUSDT 1m price action plus actual taker flow when available.",
        },
        "binance_xauusdt_order_book": (
            {
                "score": round(order_book.score, 4),
                "bid_notional": round(order_book.bid_notional, 2),
                "ask_notional": round(order_book.ask_notional, 2),
                "note": "Actual XAUUSDT futures order book; still short-lived and can be spoofed.",
            }
            if order_book
            else None
        ),
        "indicator_components_15m": {
            key: round(value, 4) for key, value in indicator_components.items()
        },
        "atr_15m": round(atr, 4),
        "legacy_backtest_signal": legacy_signal,
        "risk_plan_computed_by_code": plan_payload,
        "resistance_zone_test": (
            {
                "lower": round(resistance_test.lower, 4),
                "upper": round(resistance_test.upper, 4),
                "state": resistance_test.state,
                "touched_recently": resistance_test.touched_recently,
                "candle_pattern": resistance_test.candle_pattern,
                "rsi14": round(resistance_test.rsi14, 2),
                "ema21": round(resistance_test.ema21, 4),
                "ema50": round(resistance_test.ema50, 4),
                "ma_confluence": resistance_test.ma_confluence,
                "volume_ratio": (
                    round(resistance_test.volume_ratio, 2)
                    if resistance_test.volume_ratio is not None
                    else None
                ),
                "rejection_score_out_of_6": resistance_test.rejection_score,
                "verdict": resistance_test.verdict,
                "note": "All supplied candles were filtered to completed bars.",
            }
            if resistance_test is not None
            else None
        ),
        "breakout_retest": (
            {
                "support": [
                    round(retest.support.lower, 4),
                    round(retest.support.upper, 4),
                ],
                "resistance": [
                    round(retest.resistance.lower, 4),
                    round(retest.resistance.upper, 4),
                ],
                "long_phase": retest.long_phase,
                "short_phase": retest.short_phase,
                "long_retest_confirmed": retest.long_retest_confirmed,
                "short_retest_confirmed": retest.short_retest_confirmed,
                "actionable_side": retest.actionable_side,
                "decision_reason": retest.decision_reason,
            }
            if retest is not None
            else None
        ),
        "news_headlines_vi": [
            {"source": item.source, "title": item.title}
            for item in news_result.items
        ],
        "closed_candles_oldest_to_newest": {
            timeframe: candle_payload(frame)
            for timeframe, frame in frames.items()
        },
    }


SYSTEM_PROMPT = """
Bạn là trợ lý phân tích hợp đồng Binance Futures XAUUSDT perpetual, không phải cố vấn tài chính và không được hứa hẹn lợi nhuận.
Hãy trả lời hoàn toàn bằng tiếng Việt dựa CHỈ trên ảnh biểu đồ và JSON do ứng dụng cung cấp.

Quy tắc bắt buộc:
1. Không bịa tin, giá, đỉnh, đáy, chỉ báo, xác suất thắng hoặc dữ liệu order flow.
2. Tin tức chỉ là tiêu đề; nêu tác động theo điều kiện, không khẳng định quan hệ nhân quả.
3. Order book XAUUSDT là dữ liệu hợp đồng thật nhưng ngắn hạn và vẫn có thể bị spoof.
4. Không tự tạo hay sửa Entry, SL, TP, khối lượng hoặc đòn bẩy. Chỉ nhắc các mức code đã cung cấp.
5. Nếu deterministic_actionable=false thì stance bắt buộc là ĐỨNG NGOÀI; không được gợi ý vào sớm.
6. Nếu deterministic_actionable=true, stance chỉ được cùng dấu với deterministic_consensus_score.
7. data_consistency chỉ đo độ đồng thuận dữ liệu, KHÔNG phải xác suất thắng.
8. Mỗi trường văn bản tối đa 220 ký tự, cụ thể và tránh lặp lại.
9. Funding, open interest, order book hoặc basis không được dùng riêng lẻ để kết luận LONG/SHORT.
10. Nếu binance_futures_metrics.liquidity_guard.entries_allowed=false thì stance bắt buộc ĐỨNG NGOÀI và phải nêu rủi ro thanh khoản.
"""


def analyze_with_gemini(
    api_key: str,
    model: str,
    timeout_seconds: int,
    charts: dict[str, BytesIO] | BytesIO,
    snapshot: dict,
) -> AIMarketAnalysis:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
    )
    prompt = (
        SYSTEM_PROMPT
        + "\nDữ liệu thị trường có cấu trúc:\n"
        + json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    )
    chart_map = charts if isinstance(charts, dict) else {"15min": charts}
    model_input = [{"type": "text", "text": prompt}]
    for timeframe, chart in chart_map.items():
        model_input.extend(
            [
                {"type": "text", "text": f"Biểu đồ {timeframe}, chỉ gồm nến đã đóng:"},
                {
                    "type": "image",
                    "data": base64.b64encode(chart.getvalue()).decode("ascii"),
                    "mime_type": "image/png",
                },
            ]
        )
    interaction = client.interactions.create(
        model=model,
        store=False,
        input=model_input,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": AIMarketAnalysis.model_json_schema(),
        },
    )
    analysis = AIMarketAnalysis.model_validate_json(interaction.output_text)
    return enforce_analysis_consistency(analysis, snapshot)


def analyze_with_groq(
    api_key: str,
    model: str,
    timeout_seconds: int,
    charts: dict[str, BytesIO] | BytesIO,
    snapshot: dict,
) -> AIMarketAnalysis:
    response_template = {
        "summary_vi": "tối đa 220 ký tự",
        "chart_structure_vi": "tối đa 220 ký tự",
        "momentum_vi": "tối đa 220 ký tự",
        "news_context_vi": "tối đa 220 ký tự",
        "primary_scenario_vi": "tối đa 220 ký tự",
        "alternative_scenario_vi": "tối đa 220 ký tự",
        "risk_vi": "tối đa 220 ký tự",
        "stance": "ĐỨNG NGOÀI | NGHIÊNG LONG | NGHIÊNG SHORT",
        "data_consistency": "số nguyên 0..100",
    }
    prompt = (
        SYSTEM_PROMPT
        + "\nKhông suy luận thành tiếng, không dùng Markdown hay ```."
        + "\nChỉ trả về một JSON object hợp lệ, đủ đúng các khóa sau:\n"
        + json.dumps(response_template, ensure_ascii=False, separators=(",", ":"))
        + "\nDữ liệu thị trường có cấu trúc:\n"
        + json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    )
    chart_map = charts if isinstance(charts, dict) else {"15min": charts}
    content = [{"type": "text", "text": prompt}]
    for timeframe, chart in chart_map.items():
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Biểu đồ {timeframe}, chỉ gồm nến đã đóng:",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.b64encode(chart.getvalue()).decode("ascii")
                    },
                },
            ]
        )
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "reasoning_effort": "none",
            "temperature": 0.3,
            "top_p": 0.8,
            "max_completion_tokens": 700,
            "response_format": {"type": "json_object"},
            "stream": False,
        },
        timeout=timeout_seconds,
    )
    if not response.ok:
        raise RuntimeError(f"Groq HTTP {response.status_code}: {response.text[:500]}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Groq returned an invalid response payload") from exc
    analysis = AIMarketAnalysis.model_validate_json(content)
    return enforce_analysis_consistency(analysis, snapshot)


def enforce_analysis_consistency(
    analysis: AIMarketAnalysis,
    snapshot: dict,
) -> AIMarketAnalysis:
    if not snapshot["deterministic_actionable"]:
        analysis.stance = "ĐỨNG NGOÀI"
    elif snapshot["deterministic_consensus_score"] > 0 and analysis.stance == "NGHIÊNG SHORT":
        analysis.stance = "ĐỨNG NGOÀI"
    elif snapshot["deterministic_consensus_score"] < 0 and analysis.stance == "NGHIÊNG LONG":
        analysis.stance = "ĐỨNG NGOÀI"
    return analysis


def _complete_ai_text(value: str, limit: int) -> str:
    """Keep normal AI fields intact; only shorten pathological oversized output."""
    value = " ".join(value.split())
    if len(value) <= limit:
        return value

    sentence_end = -1
    for marker in (". ", "! ", "? ", "; "):
        position = value.rfind(marker, 0, limit)
        if position > sentence_end:
            sentence_end = position + 1
    if sentence_end >= int(limit * 0.55):
        return value[:sentence_end].rstrip()

    word_end = value.rfind(" ", 0, limit)
    if word_end >= int(limit * 0.70):
        return value[:word_end].rstrip() + "…"
    return value[:limit].rstrip() + "…"


def format_ai_analysis(
    analysis: AIMarketAnalysis,
    model: str,
    source_note: str = "phân tích mới",
) -> str:
    return "\n".join(
        [
            f"🤖 AI — {model} · {source_note}",
            f"• Kết luận: {analysis.stance} · đồng thuận {analysis.data_consistency}/100 (không phải xác suất).",
            f"• Cấu trúc 3 khung: {_complete_ai_text(analysis.chart_structure_vi, 700)}",
            f"• Nguyên nhân: {_complete_ai_text(analysis.momentum_vi + ' ' + analysis.news_context_vi, 900)}",
            f"• Điều kiện: {_complete_ai_text(analysis.primary_scenario_vi, 700)}",
            f"• Vô hiệu/rủi ro: {_complete_ai_text(analysis.alternative_scenario_vi + ' ' + analysis.risk_vi, 900)}",
        ]
    )


def build_peak_ai_snapshot(
    peak_map: PeakMap,
    gate: PeakTradeGate,
    frames: dict[str, pd.DataFrame],
    momentum_scores: dict[str, float],
    derivatives_metrics: dict,
    execution_plan: PeakExecutionPlan | None = None,
    execution_reason: str | None = None,
    liquidity: PeakLiquidityAssessment | None = None,
    hourly_structure: ChartStructure | None = None,
    daily_structure: ChartStructure | None = None,
    setup_quality=None,
    liquidity_trap=None,
    macro_risk=None,
) -> dict:
    def zone_payload(zone) -> dict | None:
        if zone is None:
            return None
        return {
            "lower": round(zone.lower, 4),
            "upper": round(zone.upper, 4),
            "reliability": zone.reliability,
            "status": zone.status,
            "timeframes": list(zone.timeframes),
            "evidence_count": zone.evidence_count,
            "reaction_atr": round(zone.reaction_atr, 2),
            "volume_spike": zone.volume_spike,
        }

    def candles(frame: pd.DataFrame) -> list[dict]:
        return [
            {
                "time": timestamp.isoformat(),
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": (
                    round(float(row["volume"]), 4)
                    if "volume" in frame.columns
                    else None
                ),
            }
            for timestamp, row in frame.tail(12).iterrows()
        ]

    return {
        "instrument": "Binance Futures XAUUSDT perpetual",
        "current_price": round(peak_map.current_price, 4),
        "deterministic_gate": {
            "allowed_decision": gate.allowed_decision,
            "reason": gate.reason,
            "long_retest_confirmed": gate.long_retest_confirmed,
            "short_rejection_confirmed": gate.short_rejection_confirmed,
            "multi_timeframe_aligned": gate.multi_timeframe_aligned,
        },
        "focus_resistance": zone_payload(gate.resistance),
        "focus_support": zone_payload(gate.support),
        "resistance_zones": [
            zone_payload(zone) for zone in peak_map.resistance_zones[:3]
        ],
        "converted_support_zones": [
            zone_payload(zone)
            for zone in peak_map.converted_support_zones[:2]
        ],
        "momentum_scores_closed_candles": {
            timeframe: round(score, 4)
            for timeframe, score in momentum_scores.items()
        },
        "binance_derivatives_metrics": derivatives_metrics,
        "one_hour_structure": (
            {
                "trend": hourly_structure.trend,
                "pattern": hourly_structure.pattern,
                "support": round(hourly_structure.support, 4),
                "resistance": round(hourly_structure.resistance, 4),
            }
            if hourly_structure is not None
            else None
        ),
        "daily_structure": (
            {
                "trend": daily_structure.trend,
                "pattern": daily_structure.pattern,
                "support": round(daily_structure.support, 4),
                "resistance": round(daily_structure.resistance, 4),
            }
            if daily_structure is not None
            else None
        ),
        "liquidity_guard": (
            {
                "is_weekend": liquidity.is_weekend,
                "status": liquidity.status,
                "volume_ratio_vs_weekday_median": (
                    round(liquidity.volume_ratio, 4)
                    if liquidity.volume_ratio is not None
                    else None
                ),
                "spread_bps": (
                    round(liquidity.spread_bps, 4)
                    if liquidity.spread_bps is not None
                    else None
                ),
                "entries_allowed": liquidity.entries_allowed,
                "reason": liquidity.reason,
            }
            if liquidity is not None
            else None
        ),
        "setup_quality_score": (
            {
                "score_out_of_100": setup_quality.score,
                "tier": setup_quality.tier,
                "recommendation": setup_quality.recommendation,
                "recommended_risk_pct": setup_quality.recommended_risk_pct,
                "actionable": setup_quality.actionable,
                "paper_only": setup_quality.paper_only,
                "factors": setup_quality.factors,
                "blockers": list(setup_quality.blockers),
                "important": "Quality score is not a win probability.",
            }
            if setup_quality is not None
            else None
        ),
        "liquidity_sweep_fomo_guard": (
            {
                "buy_side_sweep": liquidity_trap.buy_side_sweep,
                "sell_side_sweep": liquidity_trap.sell_side_sweep,
                "double_sweep": liquidity_trap.double_sweep,
                "fomo_extension": liquidity_trap.fomo_extension,
                "volume_ratio": liquidity_trap.volume_ratio,
                "latest_range_atr": liquidity_trap.latest_range_atr,
                "ema_extension_atr": liquidity_trap.ema_extension_atr,
                "reason": liquidity_trap.reason,
            }
            if liquidity_trap is not None
            else None
        ),
        "macro_event_guard": (
            {
                "blocked": macro_risk.blocked,
                "level": macro_risk.level,
                "reason": macro_risk.reason,
                "event_name": macro_risk.event_name,
                "event_time": (
                    macro_risk.event_time.isoformat()
                    if macro_risk.event_time is not None
                    else None
                ),
                "minutes_to_event": macro_risk.minutes_to_event,
                "source": macro_risk.source,
            }
            if macro_risk is not None
            else None
        ),
        "code_execution_plan": (
            {
                "side": execution_plan.side,
                "entry_zone": [
                    execution_plan.entry_lower,
                    execution_plan.entry_upper,
                ],
                "entry_reference": execution_plan.entry_reference,
                "stop_loss": execution_plan.stop_loss,
                "take_profit_1": execution_plan.take_profit_1,
                "take_profit_2": execution_plan.take_profit_2,
                "reward_risk_1": round(execution_plan.reward_risk_1, 2),
                "reward_risk_2": round(execution_plan.reward_risk_2, 2),
                "structural_target": execution_plan.structural_target,
            }
            if execution_plan is not None
            else None
        ),
        "code_execution_reason": execution_reason,
        "closed_candles_oldest_to_newest": {
            timeframe: candles(frame) for timeframe, frame in frames.items()
        },
    }


PEAK_REVIEW_SYSTEM_PROMPT = """
Bạn review bản đồ đỉnh Binance Futures XAUUSDT bằng tiếng Việt, thật ngắn và không hứa hẹn lợi nhuận.
Chỉ dùng JSON do ứng dụng cung cấp.

Quy tắc bắt buộc:
1. decision không được mạnh hơn deterministic_gate.allowed_decision. Nếu code là CHỜ thì AI bắt buộc CHỜ.
2. Không tự tạo giá, vùng, Entry, SL, TP, xác suất thắng, đòn bẩy hoặc khối lượng. Chỉ được nhắc mức trong code_execution_plan khi trường này khác null.
3. Chỉ nhắc các biên vùng có sẵn trong JSON; không khuyên đuổi giá. Nếu code_execution_plan=null thì decision bắt buộc CHỜ và nói rõ chưa được đặt lệnh.
4. Nêu rõ cần nến 15m đóng xác nhận và retest/từ chối vùng.
5. Funding, OI, basis, volume và order book không được dùng riêng lẻ để chọn hướng.
6. Mỗi trường văn bản tối đa 180 ký tự; data_consistency không phải xác suất thắng.
7. Phải ưu tiên cấu trúc 15m/1H/D1: 15m tìm Entry, 1H xác nhận, D1 không được đối nghịch mạnh. Nếu liquidity_guard.entries_allowed=false thì decision bắt buộc CHỜ.
8. setup_quality_score.score_out_of_100 là độ hoàn thiện điều kiện, tuyệt đối không diễn giải thành phần trăm thắng.
9. Nếu setup_quality_score.actionable=false, macro_event_guard.blocked=true, double_sweep=true hoặc fomo_extension=true thì decision bắt buộc CHỜ.
10. Nếu setup_quality_score.paper_only=true, mọi nhận xét CANH chỉ là paper trade; phải nói rõ không đặt lệnh thật.
"""


def enforce_peak_review_consistency(
    review: AIPeakReview,
    snapshot: dict,
) -> AIPeakReview:
    allowed = snapshot["deterministic_gate"]["allowed_decision"]
    liquidity = snapshot.get("liquidity_guard") or {}
    quality = snapshot.get("setup_quality_score") or {}
    macro = snapshot.get("macro_event_guard") or {}
    trap = snapshot.get("liquidity_sweep_fomo_guard") or {}
    if (
        allowed == "CHỜ"
        or snapshot.get("code_execution_plan") is None
        or liquidity.get("entries_allowed") is False
        or quality.get("actionable") is False
        or macro.get("blocked") is True
        or trap.get("double_sweep") is True
        or trap.get("fomo_extension") is True
    ):
        review.decision = "CHỜ"
    elif allowed == "CANH LONG" and review.decision == "CANH SHORT":
        review.decision = "CHỜ"
    elif allowed == "CANH SHORT" and review.decision == "CANH LONG":
        review.decision = "CHỜ"
    if quality.get("paper_only") is True:
        paper_notice = "PAPER ONLY — không đặt lệnh thật."
        if paper_notice not in review.risk_vi:
            review.risk_vi = f"{paper_notice} {review.risk_vi}".strip()
    return review


def analyze_peak_with_gemini(
    api_key: str,
    model: str,
    timeout_seconds: int,
    snapshot: dict,
) -> AIPeakReview:
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
    )
    prompt = (
        PEAK_REVIEW_SYSTEM_PROMPT
        + "\nDữ liệu có cấu trúc:\n"
        + json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    )
    interaction = client.interactions.create(
        model=model,
        store=False,
        input=[{"type": "text", "text": prompt}],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": AIPeakReview.model_json_schema(),
        },
    )
    review = AIPeakReview.model_validate_json(interaction.output_text)
    return enforce_peak_review_consistency(review, snapshot)


def analyze_peak_with_groq(
    api_key: str,
    model: str,
    timeout_seconds: int,
    snapshot: dict,
) -> AIPeakReview:
    template = {
        "decision": "CHỜ | CANH LONG | CANH SHORT",
        "review_vi": "tối đa 180 ký tự",
        "confirmation_vi": "tối đa 180 ký tự",
        "invalidation_vi": "tối đa 160 ký tự",
        "risk_vi": "tối đa 160 ký tự",
        "data_consistency": "số nguyên 0..100",
    }
    prompt = (
        PEAK_REVIEW_SYSTEM_PROMPT
        + "\nKhông dùng Markdown; chỉ trả một JSON object đủ các khóa sau:\n"
        + json.dumps(template, ensure_ascii=False, separators=(",", ":"))
        + "\nDữ liệu có cấu trúc:\n"
        + json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
    )
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "none",
            "temperature": 0.2,
            "top_p": 0.8,
            "max_completion_tokens": 500,
            "response_format": {"type": "json_object"},
            "stream": False,
        },
        timeout=timeout_seconds,
    )
    if not response.ok:
        raise RuntimeError(f"Groq HTTP {response.status_code}: {response.text[:500]}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Groq returned an invalid peak review payload") from exc
    review = AIPeakReview.model_validate_json(content)
    return enforce_peak_review_consistency(review, snapshot)


def format_peak_ai_review(
    review: AIPeakReview,
    model: str,
    gate: PeakTradeGate,
) -> str:
    resistance_text = (
        f"R {gate.resistance.lower:.2f}–{gate.resistance.upper:.2f}"
        if gate.resistance is not None
        else "R chưa xác định"
    )
    support_text = (
        f"S {gate.support.lower:.2f}–{gate.support.upper:.2f}"
        if gate.support is not None
        else "S chưa xác định"
    )

    return "\n".join(
        [
            f"🤖 AI REVIEW ĐỈNH — {model}",
            f"• Kết luận: {review.decision} · đồng thuận {review.data_consistency}/100 (không phải xác suất).",
            f"• Vùng code: {resistance_text} · {support_text}.",
            f"• Bộ lọc code: {_complete_ai_text(gate.reason, 700)}",
            f"• Review: {_complete_ai_text(review.review_vi, 700)}",
            f"• Xác nhận: {_complete_ai_text(review.confirmation_vi, 700)}",
            f"• Vô hiệu/rủi ro: {_complete_ai_text(review.invalidation_vi + ' ' + review.risk_vi, 900)}",
        ]
    )
