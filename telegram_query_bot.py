import logging
import os

import yaml
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, filters

from data_provider.twelvedata_provider import TwelveDataProvider
from indicators.signal_engine import MomentumBias, SignalResult, compute_momentum_bias, compute_signal

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


COMPONENT_LABEL = {"rsi": "RSI momentum", "macd": "MACD", "ema_trend": "EMA trend", "bollinger": "Bollinger %B"}


def format_reply(bias: MomentumBias, result: SignalResult, interval: str) -> str:
    # ATR duoc tinh tren khung nen live (mac dinh 15min). Dung quy tac can bac hai thoi gian
    # (random-walk scaling) de uoc luong bien dong ky vong cho 1h va 4h toi - chi la uoc luong
    # thong ke, KHONG phai cam ket huong di.
    bars_per_hour = {"1min": 60, "5min": 12, "15min": 4, "30min": 2, "45min": 1.33, "1h": 1}
    per_hour = bars_per_hour.get(interval, 4)
    move_1h = result.atr * (per_hour ** 0.5)
    move_4h = result.atr * ((per_hour * 4) ** 0.5)
    pct = bias.composite * 100

    lines = [
        f"*XAU/USD*: {bias.price:.2f} USD/oz",
        f"Xu huong ngan han: *{bias.label}* (bias {pct:+.0f}%)",
        "",
        "Chi tiet (moi chi bao tu -100% den +100%):",
    ]
    for key, val in bias.components.items():
        lines.append(f"  - {COMPONENT_LABEL[key]}: {val * 100:+.0f}%")
    lines += [
        "",
        f"Bien dong uoc luong: ~1h toi +/-{move_1h:.2f} USD, ~4h toi +/-{move_4h:.2f} USD quanh gia hien tai.",
        "",
        f"(Tham khao) Tin hieu backtest cu, it kich hoat hon: *{result.signal}* ({SIGNAL_LABEL[result.signal]})",
        "",
        "_Luu y: 'bias' la chi so xu huong ky thuat NGAN HAN dua tren RSI/MACD/EMA/Bollinger hien tai, "
        "KHONG phai du doan dam bao loi nhuan va chua duoc backtest rieng. Tin hieu backtest cu cho thay "
        "cau hinh nay chua co edge on dinh (win rate ~47%, profit factor ~0.5). Voi giao dich don bay tren "
        "Binance, luon tu dat stop-loss va khong all-in chi vi 1 con so nay._",
    ]
    return "\n".join(lines)


async def handle_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config = context.bot_data["config"]
    provider = context.bot_data["provider"]
    try:
        interval = config["live"]["interval"]
        df = provider.get_historical(interval=interval, outputsize=config["live"]["outputsize"])
        bias = compute_momentum_bias(df, config["weights"])
        result = compute_signal(
            df,
            config["weights"],
            {"buy": config["threshold_buy"], "sell": config["threshold_sell"]},
            atr_stop_multiplier=config.get("atr_stop_multiplier", 1.5),
        )
        logger.info("Signal requested by chat %s: bias=%.2f signal=%s", update.effective_chat.id, bias.composite, result.signal)
        await update.message.reply_text(format_reply(bias, result, interval), parse_mode="Markdown")
    except Exception:
        logger.exception("Failed to compute signal")
        await update.message.reply_text("Co loi khi lay gia/tinh tin hieu, thu lai sau it phut.")


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Bot tin hieu vang XAU/USD.\nGo /signal hoac /gia de xem tin hieu ky thuat + bien dong gan day."
    )


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    authorized_chat_id = int(os.environ["TELEGRAM_CHAT_ID"])
    config = load_config()
    provider = TwelveDataProvider()

    app = Application.builder().token(token).build()
    app.bot_data["config"] = config
    app.bot_data["provider"] = provider

    chat_filter = filters.Chat(chat_id=authorized_chat_id)
    app.add_handler(CommandHandler("start", handle_start, filters=chat_filter))
    app.add_handler(CommandHandler(["signal", "gia"], handle_signal, filters=chat_filter))

    logger.info("Telegram query bot started for chat_id=%s, waiting for /signal or /gia...", authorized_chat_id)
    app.run_polling()


if __name__ == "__main__":
    main()
