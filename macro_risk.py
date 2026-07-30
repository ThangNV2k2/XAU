from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import unescape
from zoneinfo import ZoneInfo

import requests


FED_FOMC_CALENDAR_URL = (
    "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
)
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; XAU-Signal-Risk-Guard/1.0; "
        "+https://www.federalreserve.gov/)"
    )
}


@dataclass(frozen=True)
class MacroEvent:
    name: str
    scheduled_at: datetime
    source: str
    importance: str = "HIGH"


@dataclass(frozen=True)
class MacroRiskAssessment:
    blocked: bool
    level: str
    reason: str
    event_name: str | None = None
    event_time: datetime | None = None
    minutes_to_event: float | None = None
    source: str | None = None
    calendar_available: bool = True


_CACHE_LOCK = threading.Lock()
_FOMC_CACHE: dict[str, object] = {
    "fetched_at": None,
    "events": [],
    "error": None,
}


def _strip_tags(value: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", value)).split())


def parse_fomc_calendar(
    page_html: str,
    decision_hour_utc: int | None = None,
    decision_hour_et: int = 14,
) -> list[MacroEvent]:
    """Parse scheduled FOMC meeting end dates from the Federal Reserve page.

    The page publishes meeting dates but not a machine-readable future release time.
    By default, attach the usual 14:00 New York statement time and convert it to UTC,
    including daylight-saving changes. A fixed UTC hour remains available for tests
    or an explicit operator override.
    """
    section_pattern = re.compile(
        r"(?P<year>20\d{2})\s+FOMC\s+Meetings(?P<body>.*?)(?="
        r"(?:20\d{2}\s+FOMC\s+Meetings)|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    meeting_pattern = re.compile(
        r"fomc-meeting__month[^>]*>\s*<strong>(?P<month>[^<]+)</strong>"
        r".*?fomc-meeting__date[^>]*>(?P<dates>.*?)</div>",
        re.IGNORECASE | re.DOTALL,
    )
    month_numbers = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    events: list[MacroEvent] = []
    for section in section_pattern.finditer(page_html):
        year = int(section.group("year"))
        for meeting in meeting_pattern.finditer(section.group("body")):
            month_name = _strip_tags(meeting.group("month")).lower()
            month = month_numbers.get(month_name)
            date_text = _strip_tags(meeting.group("dates"))
            day_numbers = [int(item) for item in re.findall(r"\d{1,2}", date_text)]
            if month is None or not day_numbers:
                continue
            end_day = day_numbers[-1]
            end_month = month
            if len(day_numbers) >= 2 and end_day < day_numbers[0]:
                end_month += 1
                if end_month == 13:
                    end_month = 1
                    year += 1
            try:
                if decision_hour_utc is None:
                    scheduled_at = datetime(
                        year,
                        end_month,
                        end_day,
                        max(0, min(23, int(decision_hour_et))),
                        tzinfo=ZoneInfo("America/New_York"),
                    ).astimezone(timezone.utc)
                else:
                    scheduled_at = datetime(
                        year,
                        end_month,
                        end_day,
                        max(0, min(23, int(decision_hour_utc))),
                        tzinfo=timezone.utc,
                    )
            except ValueError:
                continue
            events.append(
                MacroEvent(
                    name="FOMC rate decision / statement",
                    scheduled_at=scheduled_at,
                    source=FED_FOMC_CALENDAR_URL,
                )
            )
    return sorted(events, key=lambda event: event.scheduled_at)


def _manual_events(settings: dict) -> list[MacroEvent]:
    events: list[MacroEvent] = []
    for item in settings.get("manual_events", []):
        if not isinstance(item, dict) or not item.get("at_utc"):
            continue
        try:
            scheduled_at = datetime.fromisoformat(
                str(item["at_utc"]).replace("Z", "+00:00")
            )
        except ValueError:
            continue
        if scheduled_at.tzinfo is None:
            scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)
        events.append(
            MacroEvent(
                name=str(item.get("name", "High-impact macro event")),
                scheduled_at=scheduled_at.astimezone(timezone.utc),
                source=str(item.get("source", "manual config")),
                importance=str(item.get("importance", "HIGH")).upper(),
            )
        )
    return events


def fetch_fomc_events(settings: dict | None = None) -> tuple[list[MacroEvent], str | None]:
    settings = settings or {}
    cache_hours = max(1, int(settings.get("calendar_cache_hours", 6)))
    now = datetime.now(timezone.utc)
    with _CACHE_LOCK:
        fetched_at = _FOMC_CACHE.get("fetched_at")
        if (
            isinstance(fetched_at, datetime)
            and now - fetched_at < timedelta(hours=cache_hours)
        ):
            return list(_FOMC_CACHE.get("events", [])), _FOMC_CACHE.get("error")

        try:
            response = requests.get(
                settings.get("fomc_calendar_url", FED_FOMC_CALENDAR_URL),
                headers=HTTP_HEADERS,
                timeout=max(3, int(settings.get("timeout_seconds", 10))),
            )
            response.raise_for_status()
            fixed_utc_hour = settings.get("fomc_decision_hour_utc")
            events = parse_fomc_calendar(
                response.text,
                int(fixed_utc_hour) if fixed_utc_hour is not None else None,
                int(settings.get("fomc_decision_hour_et", 14)),
            )
            if not events:
                raise ValueError("Federal Reserve calendar contained no FOMC dates")
            _FOMC_CACHE.update(
                fetched_at=now,
                events=events,
                error=None,
            )
        except Exception as exc:
            # Keep a previously successful calendar if a refresh fails.
            _FOMC_CACHE["fetched_at"] = now
            _FOMC_CACHE["error"] = f"{type(exc).__name__}: {exc}"
        return list(_FOMC_CACHE.get("events", [])), _FOMC_CACHE.get("error")


def assess_macro_risk(
    analysis_now: datetime,
    settings: dict | None = None,
) -> MacroRiskAssessment:
    settings = settings or {}
    if not settings.get("enabled", True):
        return MacroRiskAssessment(False, "OFF", "Macro-event guard is disabled.")
    if analysis_now.tzinfo is None:
        analysis_now = analysis_now.replace(tzinfo=timezone.utc)
    analysis_now = analysis_now.astimezone(timezone.utc)

    events = _manual_events(settings)
    calendar_error = None
    fomc_events: list[MacroEvent] = []
    if settings.get("fetch_fomc_calendar", True):
        fomc_events, calendar_error = fetch_fomc_events(settings)
        events.extend(fomc_events)
    events = sorted(
        {
            (event.name, event.scheduled_at, event.source): event
            for event in events
        }.values(),
        key=lambda event: event.scheduled_at,
    )

    before_minutes = max(0, int(settings.get("block_minutes_before", 720)))
    after_minutes = max(0, int(settings.get("block_minutes_after", 360)))
    relevant = [
        event
        for event in events
        if event.importance == "HIGH"
        and event.scheduled_at - timedelta(minutes=before_minutes)
        <= analysis_now
        <= event.scheduled_at + timedelta(minutes=after_minutes)
    ]
    if relevant:
        event = min(
            relevant,
            key=lambda item: abs((item.scheduled_at - analysis_now).total_seconds()),
        )
        minutes = (event.scheduled_at - analysis_now).total_seconds() / 60
        position = (
            f"còn {minutes:.0f} phút"
            if minutes >= 0
            else f"đã qua {abs(minutes):.0f} phút"
        )
        return MacroRiskAssessment(
            blocked=True,
            level="HIGH",
            reason=(
                f"Khóa Entry quanh {event.name} ({position}); spread và quét hai đầu "
                "có thể tăng mạnh."
            ),
            event_name=event.name,
            event_time=event.scheduled_at,
            minutes_to_event=minutes,
            source=event.source,
            calendar_available=True,
        )

    official_calendar_missing = bool(calendar_error and not fomc_events)
    if official_calendar_missing:
        fail_closed = bool(settings.get("fail_closed_if_calendar_unavailable", False))
        return MacroRiskAssessment(
            blocked=fail_closed,
            level="UNKNOWN",
            reason=(
                "Không tải được lịch sự kiện chính thức; khóa Entry theo fail-closed."
                if fail_closed
                else "Không tải được lịch sự kiện chính thức; cần kiểm tra lịch thủ công."
            ),
            calendar_available=False,
        )
    return MacroRiskAssessment(
        False,
        "NORMAL",
        "Không nằm trong cửa sổ khóa của sự kiện HIGH đã biết.",
        calendar_available=not official_calendar_missing,
    )
