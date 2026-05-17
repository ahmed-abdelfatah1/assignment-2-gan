"""Evaluation metrics: overall CSR, per-condition match rate, diversity."""

from __future__ import annotations

from model.constraints import (
    day_of_week_token,
    is_leap,
    parse_date,
    valid_calendar_date,
    verify_date,
)
from model import config


def condition_satisfaction_rate(dates: list[str], conds_list: list[dict[str, str]]) -> float:
    """Fraction of (date, conds) pairs where every condition holds."""
    if not dates:
        return 0.0
    assert len(dates) == len(conds_list)
    n_ok = sum(1 for d, c in zip(dates, conds_list) if verify_date(d, c))
    return n_ok / len(dates)


def per_condition_breakdown(
    dates: list[str], conds_list: list[dict[str, str]]
) -> dict[str, float]:
    """Per-condition pass rate. A date that doesn't parse fails all four."""
    n = len(dates)
    if n == 0:
        return {"dow": 0.0, "mon": 0.0, "leap": 0.0, "dec": 0.0}
    counts = {"dow": 0, "mon": 0, "leap": 0, "dec": 0}
    for date_str, c in zip(dates, conds_list):
        try:
            day, month, year = parse_date(date_str)
        except ValueError:
            continue
        if not valid_calendar_date(day, month, year):
            continue
        if month - 1 == config.MONTH_TO_IDX.get(c.get("mon", ""), -1):
            counts["mon"] += 1
        try:
            if year // 10 == int(c.get("dec", "-1")):
                counts["dec"] += 1
        except ValueError:
            pass
        if (c.get("leap", "") == "True") == is_leap(year):
            counts["leap"] += 1
        if day_of_week_token(year, month, day) == c.get("dow", ""):
            counts["dow"] += 1
    return {k: v / n for k, v in counts.items()}


def diversity(samples_per_cond: list[list[str]]) -> float:
    """Mean unique-output ratio per condition group.

    Input: list where each element is the list of K samples for one condition tuple.
    Returns a number in [0, 1] (1 = every sample unique).
    """
    if not samples_per_cond:
        return 0.0
    vals: list[float] = []
    for grp in samples_per_cond:
        if not grp:
            continue
        vals.append(len(set(grp)) / len(grp))
    return sum(vals) / len(vals) if vals else 0.0
