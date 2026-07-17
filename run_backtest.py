import argparse
import json

import yaml
from dotenv import load_dotenv

from backtest.backtester import run_backtest
from backtest.data_fetcher import fetch_historical
from backtest.report import plot_equity_curve, summarize

load_dotenv()


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest gold signal engine on historical XAU/USD data")
    parser.add_argument("--interval", default="4h", choices=["1h", "4h", "1day"])
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--max-hold-bars", type=int, default=20)
    parser.add_argument(
        "--force-refresh", action="store_true", help="Re-download historical data instead of using cache"
    )
    parser.add_argument("--output", default="backtest/equity_curve.png")
    args = parser.parse_args()

    config = load_config()
    weights = config["weights"]
    thresholds = {"buy": config["threshold_buy"], "sell": config["threshold_sell"]}

    print(f"Fetching {args.years}y of {args.interval} XAU/USD history...")
    df = fetch_historical(args.interval, args.years, force_refresh=args.force_refresh)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    trades, equity_curve = run_backtest(df, weights, thresholds, max_hold_bars=args.max_hold_bars)
    stats = summarize(trades, equity_curve)

    print(json.dumps(stats, indent=2))

    if len(equity_curve) > 0:
        plot_equity_curve(equity_curve, args.output)
        print(f"Equity curve saved to {args.output}")

    if stats["total_trades"] == 0:
        print(
            "\nKhong co lenh nao duoc sinh ra trong giai doan backtest — can nhac ha threshold "
            "hoac doi khung thoi gian trong config.yaml."
        )
    elif stats["win_rate"] < 50:
        print(
            f"\nCANH BAO: Win rate {stats['win_rate']}% duoi nguong 50%. KHONG nen bat alerting "
            "that voi cau hinh nay — hay tinh chinh weights/threshold trong config.yaml roi backtest lai."
        )


if __name__ == "__main__":
    main()
