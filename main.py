"""One-shot command-line check using the same Exness scanner as Telegram."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from data_provider.exness_mt5_provider import ExnessConnectionError
from lightweight_bot import build_client, load_config
from market_scanner import ExnessMarketScanner
from strategy_engine import format_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Exness XAU/US-stock closed-candle scanner")
    parser.add_argument("--symbol", default="XAUUSD", help="XAUUSD, AAPL, NVDA, FTNT...")
    parser.add_argument("--all", action="store_true", help="Quét tất cả mã đang mở cửa")
    args = parser.parse_args()

    config = load_config()
    try:
        client = build_client(config)
    except ExnessConnectionError as exc:
        parser.exit(2, f"Lỗi Exness MT5: {exc}\n")
    scanner = ExnessMarketScanner(client, config)
    now = datetime.now(timezone.utc)
    try:
        assets = scanner.assets if args.all else [scanner.asset(args.symbol)]
        for asset in assets:
            outcome = scanner.scan_asset(asset, now, force=True)
            if outcome.analysis:
                print(format_analysis(outcome.analysis))
            else:
                print(f"{asset.symbol}: {outcome.status} · {outcome.reason}")
            print()
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
