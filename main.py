import argparse
import json
import logging
import os
from datetime import datetime

import yaml
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

from alerting.telegram_bot import TelegramAlerter
from data_provider.twelvedata_provider import TwelveDataProvider
from indicators.signal_engine import SignalResult, compute_signal

load_dotenv()

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/main.log"), logging.StreamHandler()],
)
logger = logging.getLogger("gold-signal")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_alert(result: SignalResult) -> str:
    lines = [
        f"*XAU/USD Signal: {result.signal}*",
        f"Price: {result.price:.2f}",
        f"Weighted score: {result.weighted_score:.2f}",
        f"ATR(14): {result.atr:.2f} | Suggested stop distance: {result.suggested_stop_distance:.2f}",
        "",
        "Indicators:",
    ]
    for ind in result.indicators:
        lines.append(f"  {ind.name}: value={ind.value:.2f} score={ind.score:+d}")
    lines.append(f"\nTime: {result.timestamp}")
    return "\n".join(lines)


def run_once(
    provider: TwelveDataProvider,
    config: dict,
    alerter: TelegramAlerter | None,
    dry_run: bool,
    last_signal: dict,
) -> None:
    df = provider.get_historical(interval=config["live"]["interval"], outputsize=config["live"]["outputsize"])
    weights = config["weights"]
    thresholds = {"buy": config["threshold_buy"], "sell": config["threshold_sell"]}
    result = compute_signal(df, weights, thresholds, atr_stop_multiplier=config.get("atr_stop_multiplier", 1.5))

    logger.info(json.dumps(result.to_dict()))
    print(json.dumps(result.to_dict(), indent=2))

    if result.signal != "HOLD" and result.signal != last_signal.get("signal"):
        message = format_alert(result)
        if dry_run:
            logger.info("[DRY RUN] Would send Telegram alert:\n%s", message)
        else:
            alerter.send(message)
            logger.info("Telegram alert sent.")
    last_signal["signal"] = result.signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold XAU/USD signal polling bot")
    parser.add_argument("--once", action="store_true", help="Run a single check and exit (no scheduler, no Telegram)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the polling loop but only log alerts instead of sending Telegram messages",
    )
    args = parser.parse_args()

    config = load_config()
    provider = TwelveDataProvider()
    last_signal: dict = {}

    if args.once:
        run_once(provider, config, alerter=None, dry_run=True, last_signal=last_signal)
        return

    alerter = None if args.dry_run else TelegramAlerter()

    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_once,
        "interval",
        minutes=config["live"]["poll_minutes"],
        args=[provider, config, alerter, args.dry_run, last_signal],
        next_run_time=datetime.now(),
    )
    logger.info("Starting polling loop every %s minutes...", config["live"]["poll_minutes"])
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
