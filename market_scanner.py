"""Bar-aligned Exness scanner with an in-memory multi-timeframe cache."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from data_provider.exness_mt5_provider import ExnessMT5Client
from market_sessions import market_session, timeframe_due
from strategy_engine import EMA200_WARMUP_BARS, MarketAnalysis, analyze_market


logger = logging.getLogger("exness-scanner")


INTERVAL_MINUTES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "1h": 60,
    "4h": 240,
    "1day": 1440,
}


def select_closed_candles(
    frame: pd.DataFrame,
    interval: str,
    now: datetime,
) -> pd.DataFrame:
    """Remove the currently forming MT5 bar; timestamps are bar-open times."""
    if interval not in INTERVAL_MINUTES:
        raise ValueError(f"Khung nến không hỗ trợ: {interval}")
    index = frame.index
    if index.tz is None:
        index = index.tz_localize(timezone.utc)
    else:
        index = index.tz_convert(timezone.utc)
    current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    closed = frame.copy()
    closed.index = index
    duration = pd.Timedelta(minutes=INTERVAL_MINUTES[interval])
    closed = closed.loc[closed.index + duration <= current]
    if closed.empty:
        raise ValueError(f"Chưa có nến {interval} đã đóng")
    return closed


@dataclass(frozen=True)
class AssetSpec:
    symbol: str
    name: str
    asset_type: str
    scan_minutes: int
    mode: str = "signal_only"
    enabled: bool = True


@dataclass(frozen=True)
class ScanOutcome:
    asset: AssetSpec
    analysis: MarketAnalysis | None
    refreshed_timeframes: tuple[str, ...]
    status: str
    reason: str


class ExnessMarketScanner:
    def __init__(self, client: ExnessMT5Client, config: dict) -> None:
        self.client = client
        self.config = config
        scanner = config.get("scanner", {})
        self.timeframes = tuple(
            scanner.get("timeframes", ["1min", "5min", "15min", "1h", "4h", "1day"])
        )
        self.outputsize = max(EMA200_WARMUP_BARS, int(scanner.get("bars_per_timeframe", 1000)))
        self.settle_seconds = max(0, int(scanner.get("close_settle_seconds", 8)))
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.last_bar: dict[tuple[str, str], pd.Timestamp] = {}
        self.last_daily_market_date: dict[str, object] = {}
        self.assets = self._load_assets(config)

    @staticmethod
    def _load_assets(config: dict) -> list[AssetSpec]:
        result: list[AssetSpec] = []
        for raw in config.get("assets", []):
            if not raw.get("enabled", True):
                continue
            asset_type = str(raw.get("type", "stock")).lower()
            mode = str(raw.get("mode", "signal_only")).lower()
            if mode != "signal_only":
                raise ValueError(
                    f"{raw['symbol']}: mode {mode!r} chưa được hỗ trợ; bot hiện chỉ được phép signal_only"
                )
            default_cadence = 1 if asset_type == "metal" else 5
            result.append(
                AssetSpec(
                    symbol=str(raw["symbol"]).strip().upper(),
                    name=str(raw.get("name", raw["symbol"])),
                    asset_type=asset_type,
                    scan_minutes=max(1, int(raw.get("scan_minutes", default_cadence))),
                    mode=mode,
                    enabled=True,
                )
            )
        if not result:
            raise ValueError("config.yaml chưa có assets được bật")
        return result

    def asset(self, symbol: str) -> AssetSpec:
        requested = symbol.upper().strip()
        for item in self.assets:
            if item.symbol == requested:
                return item
        raise KeyError(f"Mã {requested} không nằm trong danh sách theo dõi")

    def is_asset_due(self, asset: AssetSpec, now: datetime) -> bool:
        current = now.astimezone(timezone.utc)
        return current.minute % asset.scan_minutes == 0 and current.second >= self.settle_seconds

    def _should_refresh(self, asset: AssetSpec, interval: str, now: datetime, force: bool) -> bool:
        key = (asset.symbol, interval)
        if force or key not in self.frames:
            return True
        if interval == "1day":
            local_date = market_session(asset.asset_type, now).local_time.date()
            return self.last_daily_market_date.get(asset.symbol) != local_date
        if asset.asset_type == "stock" and interval == "1min":
            return self.is_asset_due(asset, now)
        return timeframe_due(interval, now, self.settle_seconds)

    def scan_asset(
        self,
        asset: AssetSpec,
        now: datetime | None = None,
        *,
        force: bool = False,
    ) -> ScanOutcome:
        checked_at = now or datetime.now(timezone.utc)
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=timezone.utc)
        session = market_session(asset.asset_type, checked_at)
        if not session.is_open:
            return ScanOutcome(asset, None, (), "CLOSED", session.reason)
        if not force and not self.is_asset_due(asset, checked_at):
            return ScanOutcome(asset, None, (), "NOT_DUE", "Chưa đến nhịp quét")

        refreshed: list[str] = []
        for interval in self.timeframes:
            if not self._should_refresh(asset, interval, checked_at, force):
                continue
            raw = self.client.get_rates(
                asset.symbol,
                interval,
                outputsize=self.outputsize,
            )
            closed = select_closed_candles(raw, interval, checked_at)
            key = (asset.symbol, interval)
            latest_bar = closed.index[-1]
            if force or self.last_bar.get(key) != latest_bar:
                self.frames[key] = closed.tail(self.outputsize)
                self.last_bar[key] = latest_bar
                refreshed.append(interval)
            if interval == "1day":
                self.last_daily_market_date[asset.symbol] = session.local_time.date()

        available = {
            interval: self.frames[(asset.symbol, interval)]
            for interval in self.timeframes
            if (asset.symbol, interval) in self.frames
        }
        missing = [interval for interval in self.timeframes if interval not in available]
        if missing:
            return ScanOutcome(
                asset,
                None,
                tuple(refreshed),
                "WARMING_UP",
                "Chưa có cache: " + ", ".join(missing),
            )
        if not refreshed and not force:
            return ScanOutcome(asset, None, (), "NO_NEW_BAR", "Nến mới chưa đóng")

        quote = self.client.get_quote(asset.symbol)
        if not quote.get("is_market_open", False):
            return ScanOutcome(
                asset,
                None,
                tuple(refreshed),
                "NO_QUOTE",
                "Exness chưa cho giao dịch hoặc quote chưa cập nhật",
            )
        if not quote.get("allows_new_orders", True):
            return ScanOutcome(
                asset,
                None,
                tuple(refreshed),
                "CLOSE_ONLY",
                "Exness đang ở chế độ chỉ đóng lệnh; không tạo setup mới",
            )
        quote_at = int(quote.get("last_quote_at", 0) or 0)
        quote_age = checked_at.timestamp() - quote_at if quote_at else float("inf")
        maximum_quote_age = 180 if asset.asset_type == "metal" else 600
        if quote_age > maximum_quote_age:
            return ScanOutcome(
                asset,
                None,
                tuple(refreshed),
                "STALE_QUOTE",
                f"Quote Exness đã cũ {quote_age / 60:.1f} phút; có thể nghỉ lễ/đóng cửa",
            )
        analysis = analyze_market(
            asset.symbol,
            asset.asset_type,
            available,
            quote,
            session,
            self.config.get("strategy", {}),
            checked_at,
        )
        return ScanOutcome(asset, analysis, tuple(refreshed), "OK", "Đã phân tích nến đóng")

    def scan_due(self, now: datetime | None = None) -> list[ScanOutcome]:
        checked_at = now or datetime.now(timezone.utc)
        outcomes: list[ScanOutcome] = []
        for asset in self.assets:
            if not market_session(asset.asset_type, checked_at).is_open:
                continue
            if not self.is_asset_due(asset, checked_at):
                continue
            try:
                outcomes.append(self.scan_asset(asset, checked_at))
            except Exception as exc:
                logger.exception("Scan failed for %s", asset.symbol)
                outcomes.append(
                    ScanOutcome(
                        asset,
                        None,
                        (),
                        "ERROR",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        return outcomes
