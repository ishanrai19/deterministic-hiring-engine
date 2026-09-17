"""Section detection (deterministic path)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..ingestion.loaders import Document, Token

SECTION_ALIASES: Dict[str, List[str]] = {
    "skills_section": [
        "skills", "technical skills", "core competencies", "technical competencies",
        "technologies", "tech stack", "skills summary", "key skills",
        "areas of expertise", "programming languages", "technical proficiencies",
        "competencies", "what i know", "skills and interests", "technical skillset",
    ],
    "experience": [
        "experience", "work experience", "professional experience", "employment history",
        "work history", "career history", "professional background", "employment",
        "internship", "internships", "industry experience", "jobs done", "roles held",
        "industry internship experience", "internship experience", "work exposure",
        "selected technical work",
    ],
    "project": [
        "projects", "academic projects", "personal projects", "key projects",
        "selected projects", "project experience", "project work", "research",
        "research experience", "case studies",
    ],
    "certification": [
        "certifications", "certification", "licenses and certifications",
        "licenses & certifications", "courses and certifications", "credentials",
        "licences", "licenses", "registrations", "accreditations",
    ],
    "education": [
        "education", "academic qualifications", "academic background",
        "educational qualifications", "academics", "academic details",
        "educational background", "qualifications", "training",
    ],
    "other": [
        "summary", "professional summary", "objective", "profile", "about me",
        "achievements", "awards", "publications", "languages", "interests",
        "hobbies", "references", "extra-curricular", "extracurricular", "activities",
        "contact", "relevant courses", "relevant coursework", "coursework", "courses",
        "positions of responsibility", "position of responsibility",
        "scholastic achievements", "leadership", "volunteering",
        "areas of interest", "declaration", "personal details", "memberships",
    ],
}

_HEADER_LOOKUP = {a: s for s, aliases in SECTION_ALIASES.items() for a in aliases}
_HEADER_CLEAN = re.compile(r"[^a-z& ]+")
TYPED_SECTIONS = {"skills_section", "experience", "project", "certification"}
ALL_SECTION_TYPES = TYPED_SECTIONS | {"education", "other"}


@dataclass
class Block:
    section_type: str
    text: str
    tokens: List[Token]
    start: int
    end: int
    page: int


def _normalise_header(line: str) -> str:
    return re.sub(r"\s+", " ", _HEADER_CLEAN.sub(" ", line.lower()).strip())


def _is_all_caps(line: str) -> bool:
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def classify_line(line: str) -> Tuple[Optional[str], bool]:
    stripped = line.strip()
    if not stripped or len(stripped) > 60 or len(stripped.split()) > 5:
        return None, False
    if stripped.endswith((".", ",", ";")):
        return None, False
    probe = stripped.rstrip(":").strip()
    had_colon = stripped.endswith(":")
    key = _normalise_header(probe)
    if key in _HEADER_LOOKUP:
        return _HEADER_LOOKUP[key], True
    if had_colon and len(probe.split()) <= 4:
        return None, True
    # A comma means a list ("NLP, SQL, C++"); a digit means content ("ICAR IARI, 2020").
    if _is_all_caps(probe) and "," not in probe and not re.search(r"\d", probe):
        return None, True
    return None, False


def blocks_from_spans(doc: Document, spans: List[Tuple[str, int, int]]) -> List[Block]:
    blocks: List[Block] = []
    for section, start, end in spans:
        if blocks and blocks[-1].section_type == section and start - blocks[-1].end <= 2:
            blocks[-1].end = end
        else:
            blocks.append(Block(section, "", [], start, end, 1))
    by_offset = sorted(doc.tokens, key=lambda t: t.start)
    for block in blocks:
        block.tokens = [t for t in by_offset if t.start >= block.start and t.end <= block.end]
        block.text = doc.text[block.start:block.end]
        block.page = block.tokens[0].page if block.tokens else 1
    return [b for b in blocks if b.text.strip()]


def detect_sections(doc: Document) -> List[Block]:
    lines: List[Tuple[int, int, str]] = []
    cursor = 0
    for line in doc.text.split("\n"):
        lines.append((cursor, cursor + len(line), line))
        cursor += len(line) + 1
    current = "other"
    spans: List[Tuple[str, int, int]] = []
    for start, end, line in lines:
        section, is_header = classify_line(line)
        if is_header:
            current = section if section is not None else "other"
            continue
        if line.strip():
            spans.append((current, start, end))
    return blocks_from_spans(doc, spans)


def apply_fallback(blocks: List[Block]) -> Tuple[List[Block], List[str]]:
    if any(b.section_type in TYPED_SECTIONS for b in blocks) or not blocks:
        return blocks, []
    for b in blocks:
        if b.section_type == "other":
            b.section_type = "experience"
    return blocks, ["NO_SECTIONS_DETECTED: no section headers found; skill evidence was "
                    "attributed to 'experience' by fallback and the source label is unreliable"]
