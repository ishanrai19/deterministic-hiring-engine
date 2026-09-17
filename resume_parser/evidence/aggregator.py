"""Merge mentions into Skill records, verify each bbox, build evidence counts."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import fitz

from ..extraction.skills import SkillMention
from ..schemas.candidate_profile import SOURCE_ORDER, BoundingBox, Skill, SkillLocation

_PREF = {"skills_section": 0, "experience": 1, "project": 2, "certification": 3}


def _sorted_sources(sources) -> List[str]:
    return [s for s in SOURCE_ORDER if s in sources]


def aggregate(mentions: List[SkillMention]) -> List[Skill]:
    grouped: Dict[str, List[SkillMention]] = {}
    for m in mentions:
        grouped.setdefault(m.skill_id, []).append(m)
    skills: List[Skill] = []
    for skill_id, group in grouped.items():
        sources = {m.source for m in group}
        locations, seen = [], set()
        for m in sorted(group, key=lambda x: (_PREF.get(x.source, 9), x.page)):
            key = (m.source, m.page, m.bbox)
            if key in seen:
                continue
            seen.add(key)
            locations.append(SkillLocation(
                source=m.source, page=m.page,
                bbox=BoundingBox(x0=m.bbox[0], y0=m.bbox[1], x1=m.bbox[2], y1=m.bbox[3])
                if m.bbox else None, raw_text=m.raw_text))
        best = min(group, key=lambda x: _PREF.get(x.source, 9))
        skills.append(Skill(skill_id=skill_id, skill_name=best.skill_name,
                            raw_text=best.raw_text, sources=_sorted_sources(sources),
                            evidence_count=len(sources), locations=locations))
    skills.sort(key=lambda s: s.skill_name.lower())
    return skills


def verify_locations(skills: List[Skill], pdf_path: Optional[str]) -> Tuple[List[Skill], List[str]]:
    """Re-read each bbox from the PDF and drop evidence that isn't really there."""
    warnings: List[str] = []
    if not pdf_path or not pdf_path.lower().endswith(".pdf"):
        return skills, warnings
    with fitz.open(pdf_path) as pdf:
        for skill in skills:
            kept = []
            for loc in skill.locations:
                if loc.bbox is None:
                    kept.append(loc)
                    continue
                rect = fitz.Rect(loc.bbox.x0 - 1, loc.bbox.y0 - 1,
                                 loc.bbox.x1 + 1, loc.bbox.y1 + 1)
                found = (pdf[loc.page - 1].get_textbox(rect) or "").lower().replace("\n", " ")
                probe = (loc.raw_text or skill.skill_name).lower()
                if probe[:8] in found:
                    kept.append(loc)
                else:
                    warnings.append(
                        f"SPAN_UNVERIFIED: '{skill.skill_name}' at page {loc.page} "
                        f"({loc.source}) did not re-read from its bounding box")
            skill.locations = kept
            surviving = {loc.source for loc in kept}
            if surviving and surviving != set(skill.sources):
                skill.sources = _sorted_sources(surviving)
                skill.evidence_count = len(surviving)
    return [s for s in skills if s.locations], warnings
