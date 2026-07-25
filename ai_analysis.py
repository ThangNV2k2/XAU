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
        "instrument": "XAU/USD",
        "current_price": round(current_price, 4),
        "quote_source": quote_source,
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
            "note": "Price-action proxy, not centralized XAU order flow.",
        },
        "paxg_order_book_proxy": (
            {
                "score": round(order_book.score, 4),
                "bid_notional": round(order_book.bid_notional, 2),
                "ask_notional": round(order_book.ask_notional, 2),
                "note": "PAXG/USDT proxy; can diverge from XAU/USD and can be spoofed.",
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
                "note": "Latest candle may still be forming; require a 15m close.",
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
Bạn là trợ lý phân tích thị trường XAU/USD, không phải cố vấn tài chính và không được hứa hẹn lợi nhuận.
Hãy trả lời hoàn toàn bằng tiếng Việt dựa CHỈ trên ảnh biểu đồ và JSON do ứng dụng cung cấp.

Quy tắc bắt buộc:
1. Không bịa tin, giá, đỉnh, đáy, chỉ báo, xác suất thắng hoặc dữ liệu order flow.
2. Tin tức chỉ là tiêu đề; nêu tác động theo điều kiện, không khẳng định quan hệ nhân quả.
3. PAXG order book chỉ là proxy và có thể bị spoof.
4. Không tự tạo hay sửa Entry, SL, TP, khối lượng hoặc đòn bẩy. Chỉ nhắc các mức code đã cung cấp.
5. Nếu deterministic_actionable=false thì stance bắt buộc là ĐỨNG NGOÀI; không được gợi ý vào sớm.
6. Nếu deterministic_actionable=true, stance chỉ được cùng dấu với deterministic_consensus_score.
7. data_consistency chỉ đo độ đồng thuận dữ liệu, KHÔNG phải xác suất thắng.
8. Mỗi trường văn bản tối đa 220 ký tự, cụ thể và tránh lặp lại.
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


def format_ai_analysis(
    analysis: AIMarketAnalysis,
    model: str,
    source_note: str = "phân tích mới",
) -> str:
    def compact(value: str, limit: int = 240) -> str:
        value = " ".join(value.split())
        return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"

    return "\n".join(
        [
            f"🤖 AI — {model} · {source_note}",
            f"• Kết luận: {analysis.stance} · đồng thuận {analysis.data_consistency}/100 (không phải xác suất).",
            f"• Cấu trúc 3 khung: {compact(analysis.chart_structure_vi, 160)}",
            f"• Nguyên nhân: {compact(analysis.momentum_vi + ' ' + analysis.news_context_vi, 180)}",
            f"• Điều kiện: {compact(analysis.primary_scenario_vi, 180)}",
            f"• Vô hiệu/rủi ro: {compact(analysis.alternative_scenario_vi + ' ' + analysis.risk_vi, 200)}",
        ]
    )
