"""Date normalisation and experience arithmetic. Pure deterministic code - no LLM."""

from __future__ import annotations

import re
from datetime import date
from typing import List, Optional, Tuple

MONTHS = {"jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,
          "may":5,"jun":6,"june":6,"jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,
          "september":9,"oct":10,"october":10,"nov":11,"november":11,"dec":12,"december":12}

_MON_YEAR = re.compile(r"\b(?P<mon>" + "|".join(sorted(MONTHS, key=len, reverse=True)) +
                       r")\.?\s*[-,]?\s*(?P<year>(?:19|20)\d{2}|'\d{2})\b", re.I)
_NUM_MON_YEAR = re.compile(r"\b(?P<m>0?[1-9]|1[0-2])[/-](?P<year>(?:19|20)\d{2})\b")
_YEAR_ONLY = re.compile(r"\b(?P<year>(?:19|20)\d{2})\b")
_PRESENT = re.compile(r"\b(present|current(?:ly)?|till\s+date|to\s+date|now|ongoing)\b", re.I)

# "Candidate with 10 years of experience". Deliberately requires the word
# "experience" nearby: "built a system over 3 years" is a project duration, not
# a claim of professional experience, and must not be read as one.
_STATED_YEARS = re.compile(
    r"(?:\b(?P<a>\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of\s*)?"
    r"(?:professional\s+|relevant\s+|total\s+|work\s+|industry\s+)?experience"
    r"|(?:total|overall|professional|work)\s+experience\s*(?:of|:)?\s*"
    r"\b(?P<b>\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?))", re.I)


def _year(raw: str) -> int:
    y = int(raw.strip().lstrip("'"))
    return y + 2000 if y < 100 else y


def normalise_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    raw = raw.strip()
    if _PRESENT.search(raw):
        return "Present"
    m = _MON_YEAR.search(raw)
    if m:
        return f"{_year(m.group('year')):04d}-{MONTHS[m.group('mon').lower()]:02d}"
    m = _NUM_MON_YEAR.search(raw)
    if m:
        return f"{_year(m.group('year')):04d}-{int(m.group('m')):02d}"
    m = _YEAR_ONLY.search(raw)
    if m:
        return f"{_year(m.group('year')):04d}"
    return None


def find_date_range(line: str) -> Tuple[Optional[str], Optional[str]]:
    tokens: List[Tuple[int, str]] = []
    for m in _MON_YEAR.finditer(line):
        tokens.append((m.start(), m.group()))
    if not tokens:
        for m in _NUM_MON_YEAR.finditer(line):
            tokens.append((m.start(), m.group()))
    if not tokens:
        for m in _YEAR_ONLY.finditer(line):
            tokens.append((m.start(), m.group()))
    p = _PRESENT.search(line)
    if p:
        tokens.append((p.start(), p.group()))
    tokens.sort()
    values = [v for v in (normalise_date(t) for _i, t in tokens) if v]
    if not values:
        return None, None
    return (values[0], None) if len(values) == 1 else (values[0], values[-1])


def find_stated_experience(text: str) -> Optional[float]:
    """Explicitly stated professional total. Used only when no dated employment
    exists, and never confused with a project duration."""
    values: List[float] = []
    for m in _STATED_YEARS.finditer(text or ""):
        raw = m.group("a") or m.group("b")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= value <= 60:
            values.append(value)
    return max(values) if values else None


def _to_months(value: str, *, is_end: bool, today: date) -> Optional[int]:
    if value == "Present":
        return today.year * 12 + today.month
    if re.fullmatch(r"\d{4}-\d{2}", value):
        y, m = value.split("-")
        return int(y) * 12 + int(m)
    if re.fullmatch(r"\d{4}", value):
        return int(value) * 12 + (12 if is_end else 1)
    return None


def experience_years(ranges, today: Optional[date] = None):
    today = today or date.today()
    warnings: List[str] = []
    intervals: List[Tuple[int, int]] = []
    for start, end in ranges:
        if not start:
            warnings.append("EXP_START_MISSING: employment entry without a start date")
            continue
        if not end:
            warnings.append("EXP_END_MISSING: employment entry without an end date")
            continue
        s = _to_months(start, is_end=False, today=today)
        e = _to_months(end, is_end=True, today=today)
        if s is None or e is None:
            warnings.append("EXP_DATE_UNPARSED: could not interpret an employment date")
            continue
        if e < s:
            warnings.append("EXP_RANGE_INVERTED: end date precedes start date")
            continue
        intervals.append((s, e))
    if not intervals:
        return None, warnings
    intervals.sort()
    merged: List[List[int]] = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return round(sum(e - s + 1 for s, e in merged) / 12, 1), warnings
