import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from websocket import WebSocketApp

logger = logging.getLogger("gold-query-bot.realtime")


@dataclass
class RealtimeQuote:
    price: float
    market_time: datetime
    received_at: datetime
    source: str
    is_market_open: bool
    mark_price: float | None = None
    index_price: float | None = None
    bid_price: float | None = None
    ask_price: float | None = None
    funding_rate: float | None = None
    next_funding_time: datetime | None = None
    open_interest: float | None = None


class TwelveDataPriceStream:
    def __init__(self, api_key: str, symbol: str = "XAU/USD"):
        self.api_key = api_key
        self.symbol = symbol
        self._lock = threading.Lock()
        self._latest: RealtimeQuote | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: WebSocketApp | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="twelve-data-price-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._socket:
            self._socket.close()

    def latest(self, max_received_age_seconds: int = 120) -> RealtimeQuote | None:
        with self._lock:
            quote = self._latest
        if quote is None:
            return None
        age = (datetime.now(timezone.utc) - quote.received_at).total_seconds()
        return quote if age <= max_received_age_seconds else None

    def _on_open(self, socket: WebSocketApp) -> None:
        socket.send(json.dumps({"action": "subscribe", "params": {"symbols": self.symbol}}))
        logger.info("Twelve Data WebSocket connected for %s", self.symbol)

    def _on_message(self, _socket: WebSocketApp, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
            if message.get("event") != "price" or message.get("symbol") != self.symbol:
                return
            timestamp = int(message["timestamp"])
            now = datetime.now(timezone.utc)
            quote = RealtimeQuote(
                price=float(message["price"]),
                market_time=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                received_at=now,
                source="Twelve Data WebSocket",
                is_market_open=True,
            )
            with self._lock:
                self._latest = quote
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignored malformed Twelve Data WebSocket message")

    def _on_error(self, _socket: WebSocketApp, error) -> None:
        logger.warning("Twelve Data WebSocket error: %s", error)

    def _on_close(self, _socket: WebSocketApp, status_code, message) -> None:
        logger.info("Twelve Data WebSocket closed: %s %s", status_code, message or "")

    def _run(self) -> None:
        retry_seconds = 2
        while not self._stop.is_set():
            url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}"
            self._socket = WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._socket.run_forever(ping_interval=20, ping_timeout=10)
            if self._stop.wait(retry_seconds):
                break
            retry_seconds = min(30, retry_seconds * 2)
            time.sleep(0)


class BinanceFuturesPriceStream:
    """Combined public stream for XAUUSDT execution and fair-value prices."""

    def __init__(
        self,
        symbol: str = "XAUUSDT",
        websocket_base_url: str = "wss://fstream.binance.com",
    ):
        self.symbol = symbol.upper()
        self.websocket_base_url = websocket_base_url.rstrip("/")
        self._lock = threading.Lock()
        self._latest: RealtimeQuote | None = None
        self._state: dict[str, float | int | None] = {
            "last_price": None,
            "mark_price": None,
            "index_price": None,
            "bid_price": None,
            "ask_price": None,
            "funding_rate": None,
            "next_funding_time_ms": None,
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: WebSocketApp | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="binance-futures-price-stream",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._socket:
            self._socket.close()

    def latest(self, max_received_age_seconds: int = 30) -> RealtimeQuote | None:
        with self._lock:
            quote = self._latest
        if quote is None:
            return None
        age = (datetime.now(timezone.utc) - quote.received_at).total_seconds()
        return quote if age <= max_received_age_seconds else None

    def _on_open(self, _socket: WebSocketApp) -> None:
        logger.info("Binance Futures WebSocket connected for %s", self.symbol)

    def _on_message(self, _socket: WebSocketApp, raw_message: str) -> None:
        try:
            wrapper = json.loads(raw_message)
            message = wrapper.get("data", wrapper)
            if message.get("s") != self.symbol:
                return
            event = message.get("e")
            with self._lock:
                if event == "markPriceUpdate":
                    self._state["mark_price"] = float(message["p"])
                    self._state["index_price"] = float(message["i"])
                    self._state["funding_rate"] = float(message["r"])
                    self._state["next_funding_time_ms"] = int(message["T"])
                elif event == "aggTrade":
                    self._state["last_price"] = float(message["p"])
                elif event == "bookTicker":
                    self._state["bid_price"] = float(message["b"])
                    self._state["ask_price"] = float(message["a"])
                else:
                    return

                bid = self._state["bid_price"]
                ask = self._state["ask_price"]
                mid = (
                    (float(bid) + float(ask)) / 2
                    if bid is not None and ask is not None
                    else None
                )
                price_value = (
                    self._state["last_price"]
                    or mid
                    or self._state["mark_price"]
                )
                if price_value is None:
                    return
                event_time_ms = int(
                    message.get("E")
                    or message.get("T")
                    or datetime.now(timezone.utc).timestamp() * 1000
                )
                next_funding_ms = self._state["next_funding_time_ms"]
                now = datetime.now(timezone.utc)
                self._latest = RealtimeQuote(
                    price=float(price_value),
                    market_time=datetime.fromtimestamp(
                        event_time_ms / 1000,
                        tz=timezone.utc,
                    ),
                    received_at=now,
                    source="Binance Futures WebSocket",
                    is_market_open=True,
                    mark_price=(
                        float(self._state["mark_price"])
                        if self._state["mark_price"] is not None
                        else None
                    ),
                    index_price=(
                        float(self._state["index_price"])
                        if self._state["index_price"] is not None
                        else None
                    ),
                    bid_price=float(bid) if bid is not None else None,
                    ask_price=float(ask) if ask is not None else None,
                    funding_rate=(
                        float(self._state["funding_rate"])
                        if self._state["funding_rate"] is not None
                        else None
                    ),
                    next_funding_time=(
                        datetime.fromtimestamp(
                            int(next_funding_ms) / 1000,
                            tz=timezone.utc,
                        )
                        if next_funding_ms is not None
                        else None
                    ),
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignored malformed Binance Futures WebSocket message")

    def _on_error(self, _socket: WebSocketApp, error) -> None:
        logger.warning("Binance Futures WebSocket error: %s", error)

    def _on_close(self, _socket: WebSocketApp, status_code, message) -> None:
        logger.info("Binance Futures WebSocket closed: %s %s", status_code, message or "")

    def _run(self) -> None:
        retry_seconds = 2
        stream_symbol = self.symbol.lower()
        streams = "/".join(
            [
                f"{stream_symbol}@aggTrade",
                f"{stream_symbol}@bookTicker",
                f"{stream_symbol}@markPrice@1s",
            ]
        )
        url = f"{self.websocket_base_url}/stream?streams={streams}"
        while not self._stop.is_set():
            self._socket = WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._socket.run_forever(ping_interval=20, ping_timeout=10)
            if self._stop.wait(retry_seconds):
                break
            retry_seconds = min(30, retry_seconds * 2)
            time.sleep(0)
