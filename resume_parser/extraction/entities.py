"""Education and work-history extraction from typed blocks."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..normalization.dates import find_date_range
from ..schemas.candidate_profile import Education, WorkHistory
from .sections import Block

DEGREE_PATTERNS: List[Tuple[str, str]] = [
    (r"\bm\.?\s?tech\b|\bmaster of technology\b", "M.Tech"),
    (r"\bb\.?\s?tech\b|\bbachelor of technology\b", "B.Tech"),
    (r"\bm\.?\s?e\.?\b|\bmaster of engineering\b", "M.E."),
    (r"\bb\.?\s?e\b|\bbachelor of engineering\b", "B.E."),
    (r"\bm\.?\s?sc\b|\bmaster of science\b", "M.Sc"),
    (r"\bb\.?\s?sc\b|\bbachelor of science\b", "B.Sc"),
    (r"\bmca\b|\bmaster of computer applications\b", "MCA"),
    (r"\bbca\b|\bbachelor of computer applications\b", "BCA"),
    (r"\bm\.?\s?com\b", "M.Com"),
    (r"\bb\.?\s?com\b", "B.Com"),
    (r"\bmba\b|\bmaster of business administration\b", "MBA"),
    (r"\bph\.?\s?d\b|\bdoctorate\b", "Ph.D"),
    (r"\bmasters?\b", "Masters"),
    (r"\bbachelors?\b", "Bachelors"),
    (r"\bdiploma\b", "Diploma"),
]

FIELD_AFTER_DEGREE = re.compile(
    r"(?:in|of|-)?\s*(?P<field>[A-Z][A-Za-z&/ ]{3,60}?)"
    r"(?=\s*(?:,|\||\u2013|\u2014|\s-\s|\(|/\s*[A-Z]{2,}|$))")

INSTITUTION_WORDS = re.compile(
    r"\b([A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,6}\s+"
    r"(?:University|Institute|College|School|Academy|Polytechnic)"
    r"(?:\s+of\s+[A-Z][\w.&'-]*(?:\s+[A-Z][\w.&'-]*){0,3})?"
    r"(?:\s*,\s*[A-Z][\w.'-]+)?)")

INSTITUTION_ACRONYM = re.compile(
    r"\b((?:IIT|IIIT|NIT|BITS|IIM|IISc|NID|VIT|SRM|MIT|MBIT)\b"
    r"(?:\s+[A-Z][\w.'-]*){0,3})")

YEAR = re.compile(r"\b(19[5-9]\d|20[0-9]\d)\b")
SCHOOL_ROW = re.compile(
    r"^\s*(x|xii|10th|12th|ssc|hsc|intermediate|matriculation|"
    r"senior secondary|higher secondary)\b", re.I)


def _degree_of(line: str) -> Optional[str]:
    low = line.lower()
    for pattern, canonical in DEGREE_PATTERNS:
        if re.search(pattern, low):
            return canonical
    return None


def _institution_in(line: str) -> Optional[str]:
    m = INSTITUTION_ACRONYM.search(line) or INSTITUTION_WORDS.search(line)
    return m.group(1).strip(" ,") if m else None


def _field_in(line: str, degree: str) -> Optional[str]:
    pattern = DEGREE_PATTERNS[[d for _p, d in DEGREE_PATTERNS].index(degree)][0]
    parts = re.split("(?i)" + pattern, line, maxsplit=1)
    remainder = (parts[-1] if len(parts) > 1 else line).strip(" :,-–—")
    m = FIELD_AFTER_DEGREE.match(remainder)
    if not m:
        return None
    field = re.sub(r"\s+", " ", m.group("field").strip(" ,&-"))
    return field if len(field) >= 3 and field.lower() not in {"the", "and"} else None


def _harvest(lines: List[str]) -> List[Education]:
    """On each degree line, gather institution and year from around it.

    Handles prose, institution-below-degree, and institution-ABOVE-degree.
    Backward is tried first, then forward, and each consumed line is locked -
    without the lock a degree steals its neighbour's institution and silently
    attributes a qualification to the wrong university.
    """
    entries: List[Education] = []
    degree_rows = [idx for idx, line in enumerate(lines) if _degree_of(line)]
    consumed: set = set()

    for position, i in enumerate(degree_rows):
        line = lines[i]
        if SCHOOL_ROW.match(line):
            continue
        degree = _degree_of(line)
        entry = Education(degree=degree, field=_field_in(line, degree),
                          institution=_institution_in(line))
        years = YEAR.findall(line)
        if years:
            entry.year = int(years[-1])
        consumed.add(i)

        previous = degree_rows[position - 1] if position else -1
        following = degree_rows[position + 1] if position + 1 < len(degree_rows) else len(lines)

        for scan in (range(i - 1, max(previous, i - 4), -1),
                     range(i + 1, min(following, i + 5))):
            for j in scan:
                if j < 0 or j >= len(lines) or j in consumed:
                    continue
                if SCHOOL_ROW.match(lines[j]):
                    break
                # One line can carry both ("ABC University, 2020"), so never
                # short-circuit after the institution match.
                if entry.institution is None:
                    found = _institution_in(lines[j])
                    if found:
                        entry.institution = found
                        consumed.add(j)
                if entry.year is None:
                    ys = YEAR.findall(lines[j])
                    if ys:
                        entry.year = int(ys[-1])
                        consumed.add(j)
        entries.append(entry)
    return entries


def extract_education(blocks: List[Block]) -> Tuple[List[Education], List[str]]:
    warnings: List[str] = []
    entries: List[Education] = []
    for block in blocks:
        if block.section_type == "education":
            entries.extend(_harvest([l.strip() for l in block.text.split("\n") if l.strip()]))
    if not entries:
        for block in blocks:
            if block.section_type != "other":
                continue
            entries.extend(_harvest([l.strip() for l in block.text.split("\n") if l.strip()]))
        if entries:
            warnings.append("EDUCATION_NO_HEADER: education was recovered from an "
                            "unlabelled block (degree-pattern fallback)")

    def completeness(e: Education) -> int:
        return sum(v is not None for v in (e.degree, e.field, e.institution, e.year))

    best: dict = {}
    order: List[tuple] = []
    for e in entries:
        key = (e.degree, (e.field or "").lower())
        if key not in best:
            best[key] = e
            order.append(key)
        elif completeness(e) > completeness(best[key]):
            best[key] = e
    return [best[k] for k in order], warnings


TITLE_HINTS = re.compile(
    r"\b(engineer|developer|analyst|scientist|manager|consultant|architect|intern|"
    r"administrator|specialist|lead|associate|designer|researcher|officer|executive|"
    r"programmer|tester|trainee|technician|supervisor|coordinator|instructor)\b", re.I)

COMPANY_SUFFIX = re.compile(
    r"\b(inc|inc\.|ltd|ltd\.|llc|llp|plc|pvt|private limited|limited|corp|corp\.|"
    r"corporation|technologies|technology|solutions|systems|labs|consulting|services|"
    r"software|group|gmbh|ag|co\.)\b", re.I)

SPLITTERS = re.compile(r"\s*(?:\||\u2022|\u2013|\u2014|,| at | - |\t)\s*", re.I)

BULLET_VERB = re.compile(
    r"^\s*[-•*]?\s*(collaborated|optimi[sz]ed|handled|managed|built|designed|developed|"
    r"implemented|led|created|improved|maintained|worked|performed|conducted|analy[sz]ed|"
    r"delivered|supported|automated|deployed|reduced|increased|profiled|initiated|"
    r"engineered|architected|devised|achieved|integrated|extended|applied|trained)\b", re.I)


def extract_work_history(blocks: List[Block]) -> Tuple[List[WorkHistory], List[str]]:
    entries: List[WorkHistory] = []
    warnings: List[str] = []
    for block in blocks:
        if block.section_type != "experience":
            continue
        lines = [l.strip() for l in block.text.split("\n") if l.strip()]
        for idx, line in enumerate(lines):
            if BULLET_VERB.match(line):
                continue
            if not (TITLE_HINTS.search(line) or COMPANY_SUFFIX.search(line)):
                continue
            start, end = find_date_range(line)
            parts = [p.strip(" ,|•-–—\t") for p in SPLITTERS.split(re.sub(r"\(.*?\)", " ", line))]
            parts = [p for p in parts if p]
            title = next((p for p in parts if TITLE_HINTS.search(p)), None)
            company = next((p for p in parts
                            if p != title and not TITLE_HINTS.search(p)
                            and not re.search(r"\d{4}", p)
                            and (COMPANY_SUFFIX.search(p) or p[:1].isupper())
                            and 1 <= len(p.split()) <= 6), None)
            if start is None and idx + 1 < len(lines):
                start, end = find_date_range(lines[idx + 1])
            if title is None and company is None:
                continue
            entries.append(WorkHistory(company=company,
                                       title=title.strip() if title else None,
                                       start_date=start, end_date=end))
    seen, unique = set(), []
    for e in entries:
        key = (e.company, e.title, e.start_date, e.end_date)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    if not unique:
        warnings.append("WORK_HISTORY_EMPTY: no employment entries were recognised")
    return unique, warnings
