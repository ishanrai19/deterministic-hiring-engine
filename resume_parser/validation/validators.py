"""Logical validation and parsing_status resolution."""

from __future__ import annotations

import re
from datetime import date
from typing import List, Optional

from ..normalization.dates import experience_years
from ..schemas.candidate_profile import CandidateProfile

FATAL_CODES = {"NO_TEXT_LAYER", "EMPTY_ROW", "SCHEMA_INVALID", "UNREADABLE"}
# Describe how the parse ran, not a defect in the data. These must not by
# themselves flip parsing_status, or a re-run would report differently.
INFO_CODES = {"LLM_SEGMENTATION_CACHED", "EXPERIENCE_FROM_TEXT",
              "MENTION_REJECTED", "MENTION_REDIRECTED", "PROJECTS_EMPTY"}


def _month_index(value: Optional[str], today: date) -> Optional[int]:
    if not value:
        return None
    if value == "Present":
        return today.year * 12 + today.month
    if re.fullmatch(r"\d{4}-\d{2}", value):
        y, m = value.split("-")
        return int(y) * 12 + int(m)
    if re.fullmatch(r"\d{4}", value):
        return int(value) * 12
    return None


def logical_checks(profile: CandidateProfile, today: Optional[date] = None,
                   experience_from_text: bool = False) -> List[str]:
    today = today or date.today()
    warnings: List[str] = []
    for job in profile.work_history:
        s, e = _month_index(job.start_date, today), _month_index(job.end_date, today)
        if s is not None and e is not None and e < s:
            warnings.append(f"WORK_DATES_INVERTED: {job.company or 'unknown employer'}")
        if job.start_date is None:
            warnings.append(f"WORK_START_MISSING: {job.company or 'unknown employer'}")
        if job.end_date is None:
            warnings.append(f"WORK_END_MISSING: {job.company or 'unknown employer'}")
    for edu in profile.education:
        if edu.year and edu.year > today.year + 6:
            warnings.append(f"EDU_YEAR_IMPLAUSIBLE: {edu.year}")
        if edu.degree is None:
            warnings.append("EDU_DEGREE_MISSING: education entry without a degree")
    if not experience_from_text:
        recomputed, _ = experience_years(
            [(j.start_date, j.end_date) for j in profile.work_history], today=today)
        if profile.experience_years is not None and recomputed is not None:
            if abs(profile.experience_years - recomputed) > 0.15:
                warnings.append(
                    f"EXP_MISMATCH: stated {profile.experience_years} vs recomputed {recomputed}")
    for skill in profile.skills:
        if skill.evidence_count != len(set(skill.sources)):
            warnings.append(f"EVIDENCE_COUNT_MISMATCH: {skill.skill_name}")
        if not skill.locations:
            warnings.append(f"NO_LOCATION: {skill.skill_name} has no evidence location")
    if not profile.skills:
        warnings.append("NO_SKILLS: no skill could be linked to the taxonomy")
    if not profile.education:
        warnings.append("EDUCATION_EMPTY: no education entry was recognised")
    if profile.experience_years is None:
        warnings.append("EXPERIENCE_UNKNOWN: experience_years could not be computed")
    return warnings


def resolve_status(profile: CandidateProfile, warnings: List[str]) -> str:
    codes = {w.split(":")[0] for w in warnings}
    if codes & FATAL_CODES:
        return "failed"
    if not profile.skills and not profile.work_history and not profile.education:
        return "failed"
    return "partial" if (codes - INFO_CODES) else "success"
