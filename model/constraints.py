"""Date constraint primitives — the verification core the whole pipeline rests on."""

from __future__ import annotations

import datetime as _dt
import re

from model import config

_LINE_RE = re.compile(
    r"^\[(?P<dow>[A-Z]{3})\] \[(?P<mon>[A-Z]{3})\] \[(?P<leap>True|False)\] "
    r"\[(?P<dec>\d{3})\](?:\s+(?P<date>\d{1,2}-\d{1,2}-\d{4}))?\s*$"
)

# datetime.date.weekday(): Monday == 0 ... Sunday == 6
_WD_TOKENS: tuple[str, ...] = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


def is_leap(year: int) -> bool:
    """Gregorian leap-year rule."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def day_of_week_token(year: int, month: int, day: int) -> str:
    return _WD_TOKENS[_dt.date(year, month, day).weekday()]


def parse_date(date_str: str) -> tuple[int, int, int]:
    """Parse 'd-m-yyyy' into (day, month, year). Raises ValueError on malformed input."""
    parts = date_str.strip().split("-")
    if len(parts) != 3:
        raise ValueError(f"bad date string: {date_str!r}")
    d, m, y = (int(p) for p in parts)
    return d, m, y


def parse_conds(line: str) -> dict[str, str]:
    """Parse a data.txt or example_input.txt line. Returns a conds dict (no 'date' key
    if the line is condition-only)."""
    match = _LINE_RE.match(line.rstrip("\n"))
    if not match:
        raise ValueError(f"unparseable line: {line!r}")
    out: dict[str, str] = {
        "dow": match.group("dow"),
        "mon": match.group("mon"),
        "leap": match.group("leap"),
        "dec": match.group("dec"),
    }
    if match.group("date"):
        out["date"] = match.group("date")
    return out


def valid_calendar_date(day: int, month: int, year: int) -> bool:
    """True iff (day, month, year) is a real calendar date in [1800, 2200]."""
    if year < 1800 or year > 2200:
        return False
    if month < 1 or month > 12:
        return False
    if day < 1 or day > 31:
        return False
    try:
        _dt.date(year, month, day)
    except ValueError:
        return False
    return True


def verify_date(date_str: str, conds: dict[str, str]) -> bool:
    """Return True iff date_str is a valid calendar date that satisfies all four
    input conditions (DOW, MON, LEAP, DEC)."""
    try:
        day, month, year = parse_date(date_str)
    except (ValueError, AttributeError):
        return False
    if not valid_calendar_date(day, month, year):
        return False

    expected_mon = conds.get("mon")
    if expected_mon is None or month - 1 != config.MONTH_TO_IDX.get(expected_mon, -1):
        return False

    expected_dec = conds.get("dec")
    if expected_dec is None or year // 10 != int(expected_dec):
        return False

    expected_leap = conds.get("leap")
    if expected_leap is None or (expected_leap == "True") != is_leap(year):
        return False

    expected_dow = conds.get("dow")
    if expected_dow is None or day_of_week_token(year, month, day) != expected_dow:
        return False

    return True


def which_condition_failed(date_str: str, conds: dict[str, str]) -> str:
    """For evaluation reporting: name the first failing condition.

    Returns one of 'parse', 'range', 'mon', 'dec', 'leap', 'dow', or 'ok'.
    """
    try:
        day, month, year = parse_date(date_str)
    except (ValueError, AttributeError):
        return "parse"
    if not valid_calendar_date(day, month, year):
        return "range"
    if month - 1 != config.MONTH_TO_IDX.get(conds.get("mon", ""), -1):
        return "mon"
    try:
        if year // 10 != int(conds.get("dec", "-1")):
            return "dec"
    except ValueError:
        return "dec"
    if (conds.get("leap", "") == "True") != is_leap(year):
        return "leap"
    if day_of_week_token(year, month, day) != conds.get("dow", ""):
        return "dow"
    return "ok"
