"""Trading-session rules expressed in New York time so DST is automatic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")
VIETNAM = ZoneInfo("Asia/Ho_Chi_Minh")


@dataclass(frozen=True)
class SessionState:
    is_open: bool
    phase: str
    reason: str
    local_time: datetime
    allow_new_entry: bool


def _aware(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def us_stock_session(now: datetime | None = None) -> SessionState:
    """Exness regular US stock-CFD session (09:40-15:45 New York)."""
    local = _aware(now).astimezone(NEW_YORK)
    clock = local.time().replace(tzinfo=None)
    if local.weekday() >= 5:
        return SessionState(False, "CLOSED", "Cuoi tuan", local, False)
    if clock < time(9, 40):
        return SessionState(False, "PRE_MARKET", "Chua den gio Exness mo stock CFD", local, False)
    if clock >= time(15, 45):
        return SessionState(False, "CLOSED", "Exness da dong stock CFD", local, False)
    if clock < time(9, 55):
        return SessionState(True, "OPENING_RANGE", "Dang tao bien mo cua; chua vao lenh", local, False)
    if clock < time(11, 30):
        return SessionState(True, "RETEST_WINDOW", "Uu tien breakout-retest sau mo cua", local, True)
    if clock < time(14, 0):
        return SessionState(True, "MIDDAY", "Thanh khoan giua phien thuong thap", local, False)
    if clock < time(15, 30):
        return SessionState(True, "POWER_HOUR", "Cuoi phien My", local, True)
    return SessionState(True, "CLOSE_ONLY", "Qua gan gio dong cua de mo setup moi", local, False)


def xau_session(now: datetime | None = None) -> SessionState:
    """Typical Exness XAUUSD hours, including the daily maintenance break.

    Expressing the published UTC summer/winter schedule in New York time keeps
    both sides of the DST transition stable.
    """
    local = _aware(now).astimezone(NEW_YORK)
    weekday = local.weekday()  # Monday=0, Sunday=6
    clock = local.time().replace(tzinfo=None)
    if weekday == 5:
        return SessionState(False, "CLOSED", "XAUUSD dong cua thu Bay", local, False)
    if weekday == 6:
        if clock < time(18, 5):
            return SessionState(False, "CLOSED", "XAUUSD chua mo phien Chu nhat", local, False)
        return SessionState(True, "ASIA", "XAUUSD da mo phien dau tuan", local, True)
    if weekday == 4 and clock >= time(16, 58):
        return SessionState(False, "CLOSED", "XAUUSD da dong cua cuoi tuan", local, False)
    if time(16, 58) <= clock < time(18, 1):
        return SessionState(False, "MAINTENANCE", "Exness nghi bao tri XAUUSD hang ngay", local, False)

    if time(9, 30) <= clock < time(9, 45):
        return SessionState(True, "US_OPENING_RANGE", "Dang tao bien mo phien My", local, False)
    if time(9, 45) <= clock < time(11, 30):
        return SessionState(True, "US_RETEST_WINDOW", "Uu tien breakout-retest phien My", local, True)
    if time(11, 30) <= clock < time(14, 0):
        return SessionState(True, "US_MIDDAY", "Giua phien My; loc setup chat hon", local, False)
    if time(14, 0) <= clock < time(16, 30):
        return SessionState(True, "US_POWER_HOUR", "Dong luong cuoi phien My", local, True)
    if time(16, 30) <= clock < time(16, 58):
        return SessionState(True, "CLOSE_ONLY", "Qua gan maintenance de mo setup moi", local, False)
    if 18 <= clock.hour or clock.hour < 3:
        phase = "ASIA"
    elif clock.hour < 8:
        phase = "LONDON"
    else:
        phase = "PRE_US"
    return SessionState(True, phase, "XAUUSD dang giao dich", local, True)


def market_session(asset_type: str, now: datetime | None = None) -> SessionState:
    if asset_type.lower() == "stock":
        return us_stock_session(now)
    if asset_type.lower() == "metal":
        return xau_session(now)
    raise ValueError(f"Loai tai san khong ho tro: {asset_type}")


def timeframe_due(interval: str, now: datetime, settle_seconds: int = 8) -> bool:
    """True once per aligned close minute; scanner de-duplicates actual bars."""
    current = _aware(now).astimezone(timezone.utc)
    if current.second < max(0, settle_seconds):
        return False
    minute_of_day = current.hour * 60 + current.minute
    if interval == "1min":
        return True
    if interval == "5min":
        return minute_of_day % 5 == 0
    if interval == "15min":
        return minute_of_day % 15 == 0
    if interval == "1h":
        return current.minute == 0
    if interval == "4h":
        return current.minute == 0 and current.hour % 4 == 0
    if interval == "1day":
        # Daily data is refreshed on the first active scan of a new market day;
        # the scanner also uses this signal near 00:00 UTC.
        return current.hour == 0 and current.minute == 0
    raise ValueError(f"Khung nen khong ho tro: {interval}")
