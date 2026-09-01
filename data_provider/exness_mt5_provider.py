"""Exness market data through a locally running MetaTrader 5 terminal.

The MetaTrader5 Python package talks to the desktop terminal over IPC.  This
module deliberately has no HTTP fallback: when the terminal is unavailable or
is connected to a non-Exness server, startup fails instead of mixing prices
from another venue.
"""

from __future__ import annotations

import importlib
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .base import DataProvider


INTERVAL_ATTRIBUTES = {
    "1min": "TIMEFRAME_M1",
    "5min": "TIMEFRAME_M5",
    "15min": "TIMEFRAME_M15",
    "1h": "TIMEFRAME_H1",
    "4h": "TIMEFRAME_H4",
    "1day": "TIMEFRAME_D1",
}


class ExnessConnectionError(RuntimeError):
    """Raised when an Exness MT5 connection or request cannot be completed."""


def _optional_int(value: str | int | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _utc(value: str | datetime) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(timezone.utc)
    else:
        timestamp = timestamp.tz_convert(timezone.utc)
    return timestamp.to_pydatetime()


def _normalize_symbol(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


class ExnessMT5Client:
    """Thread-safe, read-only market-data client for one Exness MT5 terminal."""

    def __init__(
        self,
        *,
        path: str | None = None,
        login: int | str | None = None,
        password: str | None = None,
        server: str | None = None,
        timeout_ms: int = 15_000,
        portable: bool = False,
        require_exness_server: bool = True,
        mt5_module: Any | None = None,
        auto_connect: bool = True,
    ) -> None:
        self.path = (path or os.getenv("EXNESS_MT5_PATH", "")).strip() or None
        self.login = _optional_int(login or os.getenv("EXNESS_MT5_LOGIN"))
        self.password = password or os.getenv("EXNESS_MT5_PASSWORD") or None
        self.server = server or os.getenv("EXNESS_MT5_SERVER") or None
        self.timeout_ms = max(1_000, int(timeout_ms))
        self.portable = bool(portable)
        self.require_exness_server = bool(require_exness_server)
        self._mt5 = mt5_module
        self._lock = threading.RLock()
        self._connected = False
        self._symbol_cache: dict[str, str] = {}
        if auto_connect:
            self.connect()

    @property
    def mt5(self):
        if self._mt5 is None:
            try:
                self._mt5 = importlib.import_module("MetaTrader5")
            except ImportError as exc:
                raise ExnessConnectionError(
                    "Chua cai package MetaTrader5. Chay: pip install MetaTrader5"
                ) from exc
        return self._mt5

    def connect(self) -> None:
        with self._lock:
            if self._connected:
                return
            kwargs: dict[str, Any] = {
                "timeout": self.timeout_ms,
                "portable": self.portable,
            }
            if self.path:
                kwargs["path"] = self.path
            if self.login is not None:
                kwargs["login"] = self.login
            if self.password:
                kwargs["password"] = self.password
            if self.server:
                kwargs["server"] = self.server
            if not self.mt5.initialize(**kwargs):
                raise ExnessConnectionError(
                    f"Khong ket noi duoc terminal MT5: {self.mt5.last_error()}"
                )
            account = self.mt5.account_info()
            if account is None:
                self.mt5.shutdown()
                raise ExnessConnectionError(
                    "MT5 chua dang nhap tai khoan; hay dang nhap Exness trong terminal."
                )
            identity = " ".join(
                str(getattr(account, field, ""))
                for field in ("server", "company")
            )
            if self.require_exness_server and "exness" not in identity.lower():
                self.mt5.shutdown()
                raise ExnessConnectionError(
                    f"Terminal dang ket noi server khong phai Exness: {identity.strip()}"
                )
            self._connected = True

    def shutdown(self) -> None:
        with self._lock:
            if self._connected:
                self.mt5.shutdown()
                self._connected = False

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def resolve_symbol(self, requested: str) -> str:
        """Resolve account-specific suffixes such as ``XAUUSDm`` safely."""
        key = requested.upper().strip()
        if key in self._symbol_cache:
            return self._symbol_cache[key]
        with self._lock:
            self._ensure_connected()
            symbols = self.mt5.symbols_get()
            if symbols is None:
                raise ExnessConnectionError(
                    f"Khong doc duoc danh sach symbol: {self.mt5.last_error()}"
                )
            requested_normalized = _normalize_symbol(requested)
            exact = [item for item in symbols if item.name.upper() == key]
            prefixed = [
                item
                for item in symbols
                if _normalize_symbol(item.name).startswith(requested_normalized)
            ]
            candidates = exact or prefixed
            if not candidates:
                raise ExnessConnectionError(
                    f"Tai khoan Exness khong co ma {requested}. Kiem tra Market Watch."
                )
            # Prefer an already visible/tradable instrument and the shortest suffix.
            candidates.sort(
                key=lambda item: (
                    not bool(getattr(item, "visible", False)),
                    int(getattr(item, "trade_mode", 0)) == 0,
                    len(item.name),
                )
            )
            resolved = str(candidates[0].name)
            if not self.mt5.symbol_select(resolved, True):
                raise ExnessConnectionError(
                    f"Khong bat duoc {resolved} trong Market Watch: {self.mt5.last_error()}"
                )
            self._symbol_cache[key] = resolved
            return resolved

    def get_rates(
        self,
        symbol: str,
        interval: str,
        outputsize: int = 260,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if interval not in INTERVAL_ATTRIBUTES:
            raise ValueError(f"Khung nen Exness MT5 khong ho tro: {interval}")
        if not 1 <= int(outputsize) <= 50_000:
            raise ValueError("outputsize phai nam trong 1..50000")
        with self._lock:
            self._ensure_connected()
            resolved = self.resolve_symbol(symbol)
            timeframe = getattr(self.mt5, INTERVAL_ATTRIBUTES[interval])
            if start_date and end_date:
                start = _utc(start_date)
                end = _utc(end_date)
                rows = self.mt5.copy_rates_range(resolved, timeframe, start, end)
            elif start_date or end_date:
                anchor = _utc(end_date or start_date)
                rows = self.mt5.copy_rates_from(
                    resolved,
                    timeframe,
                    anchor,
                    int(outputsize),
                )
            else:
                # Position zero is the forming bar. Request one extra so callers can
                # remove it without losing a completed warm-up bar.
                rows = self.mt5.copy_rates_from_pos(
                    resolved,
                    timeframe,
                    0,
                    int(outputsize) + 1,
                )
            if rows is None or len(rows) == 0:
                raise ExnessConnectionError(
                    f"Exness MT5 khong tra nen {resolved} {interval}: {self.mt5.last_error()}"
                )
        frame = pd.DataFrame(rows)
        frame["datetime"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        frame = frame.set_index("datetime").sort_index()
        rename = {"tick_volume": "volume"}
        frame = frame.rename(columns=rename)
        numeric = [
            name
            for name in ("open", "high", "low", "close", "volume", "spread", "real_volume")
            if name in frame.columns
        ]
        for column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        required = ["open", "high", "low", "close"]
        if frame[required].isna().any().any():
            frame = frame.dropna(subset=required)
        columns = required + [
            name for name in ("volume", "spread", "real_volume") if name in frame.columns
        ]
        return frame[columns].tail(int(outputsize) + 1)

    def get_quote(self, symbol: str) -> dict:
        with self._lock:
            self._ensure_connected()
            resolved = self.resolve_symbol(symbol)
            tick = self.mt5.symbol_info_tick(resolved)
            info = self.mt5.symbol_info(resolved)
            if tick is None or info is None:
                raise ExnessConnectionError(
                    f"Exness MT5 khong tra quote {resolved}: {self.mt5.last_error()}"
                )
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        last = float(getattr(tick, "last", 0.0) or 0.0)
        close = last if last > 0 else (bid + ask) / 2 if bid > 0 and ask > 0 else bid or ask
        trade_mode = int(getattr(info, "trade_mode", 0))
        return {
            "symbol": resolved,
            "requested_symbol": symbol,
            "close": close,
            "bid": bid or None,
            "ask": ask or None,
            "last_quote_at": int(getattr(tick, "time", 0) or 0),
            "is_market_open": bool(close > 0 and trade_mode != 0),
            "allows_new_orders": trade_mode in {1, 2, 4},
            "trade_mode": trade_mode,
            "source": f"Exness MT5 ({getattr(info, 'path', resolved)})",
            "digits": int(getattr(info, "digits", 2)),
            "point": float(getattr(info, "point", 0.01)),
            "volume_min": float(getattr(info, "volume_min", 0.0)),
            "volume_step": float(getattr(info, "volume_step", 0.0)),
        }


class ExnessMT5Provider(DataProvider):
    """Single-symbol adapter kept compatible with the project's provider API."""

    def __init__(
        self,
        symbol: str = "XAUUSD",
        client: ExnessMT5Client | None = None,
        **client_kwargs,
    ) -> None:
        self.symbol = symbol
        self.client = client or ExnessMT5Client(**client_kwargs)

    @property
    def resolved_symbol(self) -> str:
        return self.client.resolve_symbol(self.symbol)

    def get_historical(
        self,
        interval: str,
        outputsize: int = 100,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.client.get_rates(
            self.symbol,
            interval,
            outputsize,
            start_date,
            end_date,
        )

    def get_latest_price(self) -> float:
        return float(self.get_quote()["close"])

    def get_quote(self) -> dict:
        return self.client.get_quote(self.symbol)

    def shutdown(self) -> None:
        self.client.shutdown()
