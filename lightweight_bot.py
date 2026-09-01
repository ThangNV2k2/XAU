"""Lightweight Telegram bot: Exness candles in, deterministic alerts out."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, filters

from data_provider.exness_mt5_provider import ExnessConnectionError, ExnessMT5Client
from market_scanner import AssetSpec, ExnessMarketScanner, ScanOutcome
from market_sessions import market_session
from strategy_engine import MarketAnalysis, format_analysis


load_dotenv()
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            "logs/exness_scanner.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("apscheduler").setLevel(logging.WARNING)
logger = logging.getLogger("exness-light-bot")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class AlertState:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.data = {"signals": {}, "errors": {}}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data.update(loaded)
        except (OSError, ValueError, TypeError):
            pass

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    @staticmethod
    def _signal_fingerprint(analysis: MarketAnalysis) -> str:
        plan = analysis.plan
        if plan is None:
            return ""
        return "|".join(
            [
                analysis.symbol,
                plan.side,
                plan.setup,
                f"{plan.preferred_entry:.8g}",
                f"{plan.stop_loss:.8g}",
            ]
        )

    def signal_is_new(self, analysis: MarketAnalysis, cooldown_minutes: int) -> bool:
        if analysis.plan is None:
            return False
        key = analysis.symbol
        fingerprint = self._signal_fingerprint(analysis)
        previous = self.data.setdefault("signals", {}).get(key, {})
        previous_at = _parse_time(previous.get("at"))
        if previous.get("fingerprint") == fingerprint:
            return False
        if previous_at and analysis.checked_at - previous_at < timedelta(minutes=cooldown_minutes):
            return False
        self.data["signals"][key] = {
            "fingerprint": fingerprint,
            "at": analysis.checked_at.isoformat(),
        }
        self.save()
        return True

    def error_is_due(self, key: str, now: datetime, cooldown_minutes: int = 60) -> bool:
        previous = _parse_time(self.data.setdefault("errors", {}).get(key))
        if previous and now - previous < timedelta(minutes=cooldown_minutes):
            return False
        self.data["errors"][key] = now.isoformat()
        self.save()
        return True


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result if result.tzinfo else result.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _seconds_to_next_minute(settle_seconds: int) -> float:
    now = datetime.now(timezone.utc)
    boundary = now.replace(second=0, microsecond=0) + timedelta(minutes=1, seconds=settle_seconds)
    return max(1.0, (boundary - now).total_seconds())


def _message_chunks(header: str, sections: list[str], limit: int = 4096) -> list[str]:
    """Keep grouped Telegram alerts intact without silent 4096-char truncation."""
    chunks: list[str] = []
    current = header
    for section in sections:
        addition = "\n\n" + section
        if len(current) + len(addition) <= limit:
            current += addition
            continue
        if current != header:
            chunks.append(current)
        # Compact analyses are normally far below the Telegram limit. Keep a
        # defensive hard cap so one malformed symbol can never block the rest.
        current = header + "\n\n" + section[: max(0, limit - len(header) - 2)]
    if current != header:
        chunks.append(current)
    return chunks


def journal_signal(analysis: MarketAnalysis, path: str = "logs/signal_journal.jsonl") -> None:
    """Ghi mỗi tín hiệu Entry đã phát ra một dòng JSON.

    Tin nhắn Telegram trôi đi thì không còn gì để chấm điểm về sau. Nhật ký này
    là thứ duy nhất cho phép đo hiệu quả forward: mỗi dòng đủ để tính lại R khi
    đối chiếu với giá thật sau đó. Lỗi ghi file không bao giờ được phép làm
    hỏng vòng quét, nên mọi OSError bị nuốt và chỉ log lại.
    """
    plan = analysis.plan
    if plan is None:
        return
    record = {
        "at": analysis.checked_at.isoformat(),
        "symbol": analysis.symbol,
        "asset_type": analysis.asset_type,
        "phase": analysis.market_phase,
        "action": analysis.action,
        "side": plan.side,
        "setup": plan.setup,
        "entry": plan.preferred_entry,
        "entry_lower": plan.entry_lower,
        "entry_upper": plan.entry_upper,
        "stop_loss": plan.stop_loss,
        "take_profit_1": plan.take_profit_1,
        "take_profit_2": plan.take_profit_2,
        "invalidation": plan.invalidation_level,
        "risk": plan.risk,
        "bias": analysis.bias,
        "confidence": analysis.confidence,
        "intraday_score": round(analysis.intraday_score, 4),
        "long_term_score": round(analysis.long_term_score, 4),
    }
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.exception("Không ghi được nhật ký tín hiệu vào %s", path)


def _entry_alert_title(asset: AssetSpec) -> str:
    if asset.asset_type == "stock":
        return "📣 CỔ PHIẾU · CHỈ THÔNG BÁO"
    return "🔔 XAU · TÍN HIỆU RETEST"


def _scanner(context: ContextTypes.DEFAULT_TYPE) -> ExnessMarketScanner:
    return context.bot_data["scanner"]


def _authorized(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    return bool(update.effective_chat and update.effective_chat.id == context.bot_data["chat_id"])


async def _send_outcome(update: Update, outcome: ScanOutcome) -> None:
    if outcome.analysis is not None:
        await update.effective_message.reply_text(format_analysis(outcome.analysis))
    else:
        state = market_session(outcome.asset.asset_type, datetime.now(timezone.utc))
        await update.effective_message.reply_text(
            f"{outcome.asset.symbol}: {outcome.status} · {outcome.reason}\n"
            f"Giờ New York: {state.local_time:%Y-%m-%d %H:%M} · bot không gọi dữ liệu khi thị trường đóng."
        )


async def handle_xau(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update, context):
        return
    scanner = _scanner(context)
    metal = next((asset for asset in scanner.assets if asset.asset_type == "metal"), None)
    if metal is None:
        await update.effective_message.reply_text("Tài khoản Exness hiện không có XAUUSD khả dụng.")
        return
    try:
        outcome = await asyncio.to_thread(scanner.scan_asset, metal, datetime.now(timezone.utc), force=True)
    except Exception as exc:
        await update.effective_message.reply_text(f"Không đọc được Exness: {type(exc).__name__}: {exc}")
        return
    await _send_outcome(update, outcome)


async def handle_stock(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update, context):
        return
    scanner = _scanner(context)
    symbol = context.args[0].upper() if context.args else ""
    stocks = [asset for asset in scanner.assets if asset.asset_type == "stock"]
    if not symbol:
        status = market_session("stock", datetime.now(timezone.utc))
        await update.effective_message.reply_text(
            "Cổ phiếu lớn đang theo dõi: " + ", ".join(asset.symbol for asset in stocks) + "\n"
            f"Phiên: {status.phase} · {status.reason}\n"
            "Dùng /cp AAPL (hoặc NVDA, FTNT...) để phân tích một mã."
        )
        return
    try:
        asset = scanner.asset(symbol)
        if asset.asset_type != "stock":
            raise KeyError(symbol)
        outcome = await asyncio.to_thread(scanner.scan_asset, asset, datetime.now(timezone.utc), force=True)
    except KeyError:
        await update.effective_message.reply_text(f"Mã {symbol} không nằm trong nhóm cổ phiếu lớn đã cấu hình.")
        return
    except Exception as exc:
        await update.effective_message.reply_text(f"Không đọc được {symbol} từ Exness: {type(exc).__name__}: {exc}")
        return
    await _send_outcome(update, outcome)


async def handle_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update, context):
        return
    scanner = _scanner(context)
    now = datetime.now(timezone.utc)
    session = market_session("stock", now)
    if not session.is_open:
        await update.effective_message.reply_text(
            f"Cổ phiếu Mỹ chưa active: {session.reason} · New York {session.local_time:%H:%M}."
        )
        return
    await update.effective_message.reply_text("Đang đọc nến Exness cho nhóm cổ phiếu lớn…")
    messages = []
    for asset in [item for item in scanner.assets if item.asset_type == "stock"]:
        try:
            outcome = await asyncio.to_thread(scanner.scan_asset, asset, now, force=True)
            if outcome.analysis:
                messages.append(format_analysis(outcome.analysis, compact=True))
        except Exception as exc:
            messages.append(f"⚠ {asset.symbol}: {type(exc).__name__}: {exc}")
    if not messages:
        await update.effective_message.reply_text("Chưa có phân tích.")
        return
    for message in _message_chunks("📋 US STOCKS · CHỈ THÔNG BÁO", messages):
        await update.effective_message.reply_text(message)


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _authorized(update, context):
        return
    now = datetime.now(timezone.utc)
    xau = market_session("metal", now)
    stocks = market_session("stock", now)
    scanner = _scanner(context)
    await update.effective_message.reply_text(
        "EXNESS SCANNER\n"
        f"• XAUUSD: {'ACTIVE' if xau.is_open else 'OFF'} · {xau.phase} · {xau.reason}\n"
        f"• US stocks: {'ACTIVE' if stocks.is_open else 'OFF'} · {stocks.phase} · {stocks.reason}\n"
        f"• Cache: {len(scanner.frames)} khung · nguồn duy nhất Exness MT5\n"
        "• Telegram: XAU + US stocks · Entry/4H/D1\n"
        "• Chế độ: signal_only, không gửi lệnh MT5\n"
        "• Nhịp: XAU 1 phút; stocks 5 phút; chỉ cập nhật khung đến hạn."
    )


def help_text() -> str:
    return (
        "BOT EXNESS ĐA KHUNG\n"
        "• /xau hoặc /gia — XAUUSD: RSI, xu hướng, retest, Entry/SL/TP\n"
        "• /cp AAPL — phân tích một cổ phiếu lớn\n"
        "• /stocks — bản quét gọn cả danh sách cổ phiếu\n"
        "• /status — giờ giao dịch và trạng thái cache\n\n"
        "XAU và stocks đều tự gửi Entry/4H/D1 về Telegram. Cổ phiếu chỉ thông báo.\n"
        "Bot không đặt lệnh. BUY/SHORT chỉ xuất hiện sau nến đóng retest; ngoài giờ giao dịch không gọi dữ liệu."
    )


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if _authorized(update, context):
        await update.effective_message.reply_text(help_text())


async def scan_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    scanner = _scanner(context)
    state: AlertState = context.bot_data["alert_state"]
    now = datetime.now(timezone.utc)
    settings = context.bot_data["config"].get("alerts", {})
    if not settings.get("telegram_enabled", True):
        return
    outcomes = await asyncio.to_thread(scanner.scan_due, now)
    horizon_4h: list[str] = []
    horizon_daily: list[str] = []
    cooldown = int(settings.get("signal_cooldown_minutes", 60))
    for outcome in outcomes:
        if outcome.status == "ERROR":
            if state.error_is_due(outcome.asset.symbol, now):
                await context.bot.send_message(
                    chat_id=context.bot_data["chat_id"],
                    text=f"⚠ Lỗi Exness {outcome.asset.symbol}: {outcome.reason}",
                )
            continue
        analysis = outcome.analysis
        if analysis is None:
            continue
        if settings.get("entry_alerts", True) and state.signal_is_new(analysis, cooldown):
            journal_signal(analysis, settings.get("journal_path", "logs/signal_journal.jsonl"))
            await context.bot.send_message(
                chat_id=context.bot_data["chat_id"],
                text=_entry_alert_title(outcome.asset) + "\n" + format_analysis(analysis),
            )
        summary = format_analysis(analysis, compact=True)
        if "1day" in outcome.refreshed_timeframes:
            horizon_daily.append(summary)
        if "4h" in outcome.refreshed_timeframes:
            horizon_4h.append(summary)
    if horizon_daily and settings.get("daily_summary", True):
        for message in _message_chunks("🗓 DỰ BÁO D1 · XAU + STOCKS · nến Exness đã đóng", horizon_daily):
            await context.bot.send_message(chat_id=context.bot_data["chat_id"], text=message)
    if horizon_4h and settings.get("four_hour_summary", True):
        for message in _message_chunks("📊 DỰ BÁO 4H · XAU + STOCKS · nến Exness đã đóng", horizon_4h):
            await context.bot.send_message(chat_id=context.bot_data["chat_id"], text=message)


async def startup_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    scanner = _scanner(context)
    unavailable: list[str] = []
    valid: list[AssetSpec] = []
    for asset in scanner.assets:
        try:
            await asyncio.to_thread(scanner.client.resolve_symbol, asset.symbol)
            valid.append(asset)
        except Exception as exc:
            unavailable.append(f"{asset.symbol} ({exc})")
    scanner.assets = valid
    text = (
        "✅ EXNESS SCANNER ĐÃ BẬT\n"
        "Telegram: XAU + cổ phiếu Mỹ · Entry/4H/D1.\n"
        "XAU quét theo phút; cổ phiếu theo 5 phút và chỉ thông báo.\n"
        "Chế độ signal_only: không gửi lệnh MT5.\n"
        "Ngoài giờ giao dịch: không gọi dữ liệu, không sinh tín hiệu."
    )
    if unavailable:
        text += "\n⚠ Mã không có trên tài khoản Exness nên đã bỏ qua: " + "; ".join(unavailable)
    await context.bot.send_message(chat_id=context.bot_data["chat_id"], text=text[:4096])


def build_client(config: dict) -> ExnessMT5Client:
    settings = config.get("exness", {})
    return ExnessMT5Client(
        path=settings.get("terminal_path") or None,
        timeout_ms=int(settings.get("timeout_ms", 15_000)),
        portable=bool(settings.get("portable", False)),
        require_exness_server=bool(settings.get("require_exness_server", True)),
    )


def main() -> None:
    config = load_config()
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = int(os.environ["TELEGRAM_CHAT_ID"])
    try:
        client = build_client(config)
    except ExnessConnectionError as exc:
        raise SystemExit(f"Lỗi Exness MT5: {exc}") from exc
    scanner = ExnessMarketScanner(client, config)
    app = Application.builder().token(token).build()
    app.bot_data.update(
        {
            "config": config,
            "chat_id": chat_id,
            "client": client,
            "scanner": scanner,
            "alert_state": AlertState(config.get("alerts", {}).get("state_path", "logs/exness_alert_state.json")),
        }
    )
    chat_filter = filters.Chat(chat_id=chat_id)
    app.add_handler(CommandHandler(["start", "h", "help"], handle_help, filters=chat_filter))
    app.add_handler(CommandHandler(["xau", "gia", "signal"], handle_xau, filters=chat_filter))
    app.add_handler(CommandHandler(["cp", "stock"], handle_stock, filters=chat_filter))
    app.add_handler(CommandHandler("stocks", handle_stocks, filters=chat_filter))
    app.add_handler(CommandHandler("status", handle_status, filters=chat_filter))
    settle = int(config.get("scanner", {}).get("close_settle_seconds", 8))
    app.job_queue.run_repeating(
        scan_job,
        interval=60,
        first=_seconds_to_next_minute(settle),
        name="exness-closed-candle-scanner",
    )
    app.job_queue.run_once(startup_job, when=2, name="exness-startup-check")
    logger.info("Starting Exness scanner for %s", ", ".join(asset.symbol for asset in scanner.assets))
    try:
        app.run_polling()
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
