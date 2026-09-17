"""Stage 3 - skill/entity MENTION EXTRACTION.

This is a gazetteer scan, not a learned NER model: it finds occurrences of terms
already present in the taxonomy. That trades recall for precision, which is the
right trade here, and it is why taxonomy coverage - not model quality - is the
limiting factor on how many skills get found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..ingestion.loaders import Token
from ..normalization.taxonomy import SkillTaxonomy
from .sections import Block

_EDGE_JUNK = re.compile(r"^[\s:;,\.\|/•\*\-–—\(\)\[\]]+|[\s:;,\.\|/•\*\-–—\(\)\[\]]+$")
_HYPHEN_TAIL = re.compile(r"^(?P<stem>[A-Za-z]{2,})-$")
_LABEL_PREFIX = re.compile(
    r"^\s*(programming languages?|languages?|web tech(?:nologies)?(?:\s*&\s*database)?|"
    r"databases?|machine learning|deep learning|networking tools?|core competencies|"
    r"tools?|frameworks?|libraries|technologies|others?|coursework|skills learnt|"
    r"technical skills?|skills?)\s*:", re.I)


def _clean(text: str) -> str:
    return _EDGE_JUNK.sub("", text).strip()


@dataclass(frozen=True)
class SkillMention:
    skill_id: str
    skill_name: str
    raw_text: str
    source: str
    page: int
    bbox: Optional[Tuple[float, float, float, float]]
    match_rung: str
    char_start: int = 0
    char_end: int = 0


def _line_groups(tokens: List[Token]) -> List[Tuple[tuple, List[Token]]]:
    groups: Dict[tuple, List[Token]] = {}
    for t in tokens:
        groups.setdefault(t.line_key, []).append(t)
    for g in groups.values():
        g.sort(key=lambda t: t.start)
    return sorted(groups.items(), key=lambda kv: min(t.start for t in kv[1]))


def _bbox(tokens: List[Token], has_layout: bool):
    if not has_layout:
        return None
    return (min(t.x0 for t in tokens), min(t.y0 for t in tokens),
            max(t.x1 for t in tokens), max(t.y1 for t in tokens))


def _dewrap(groups) -> List[List[Token]]:
    """Join a hyphen-broken word onto the next line's first token.

    The merged token keeps the stem's geometry, so its bbox still re-reads.
    """
    lines = [list(tokens) for _key, tokens in groups]
    for i in range(len(lines) - 1):
        if not lines[i] or not lines[i + 1]:
            continue
        last = lines[i][-1]
        match = _HYPHEN_TAIL.match(last.text)
        if not match or not lines[i + 1][0].text[:1].islower():
            continue
        nxt = lines[i + 1][0]
        lines[i][-1] = Token(match.group("stem") + nxt.text, last.page, last.x0, last.y0,
                             last.x1, last.y1, last.line_key, last.start, nxt.end)
        lines[i + 1] = lines[i + 1][1:]
    return [l for l in lines if l]


def _strip_label(tokens: List[Token]) -> List[Token]:
    """Drop a leading 'Something:' label so it is not scanned as a skill."""
    text = " ".join(t.text for t in tokens)
    match = _LABEL_PREFIX.match(text)
    if not match:
        return tokens
    consumed = 0
    for idx, token in enumerate(tokens):
        consumed += len(token.text) + (1 if idx else 0)
        if consumed >= match.end():
            return tokens[idx + 1:]
    return tokens


def scan_block(block: Block, taxonomy: SkillTaxonomy, has_layout: bool = True):
    """Longest-match-first n-gram scan, line by line."""
    mentions: List[SkillMention] = []
    for line_tokens in _dewrap(_line_groups(block.tokens)):
        if block.section_type == "skills_section":
            line_tokens = _strip_label(line_tokens)
        i, n = 0, len(line_tokens)
        while i < n:
            found = None
            for size in taxonomy.surface_form_lengths():
                if i + size > n:
                    continue
                window = line_tokens[i:i + size]
                phrase = _clean(" ".join(t.text for t in window))
                if phrase and taxonomy.lookup_surface(phrase):
                    found = (window, phrase)
                    break
            if found:
                window, phrase = found
                match = taxonomy.link(phrase, original=phrase)
                if match:
                    mentions.append(SkillMention(
                        match.skill_id, match.skill_name, phrase, block.section_type,
                        window[0].page, _bbox(window, has_layout), match.match_rung,
                        window[0].start, window[-1].end))
                i += len(window)
            else:
                i += 1
    return mentions


def scan_certifications(block: Block, taxonomy: SkillTaxonomy, has_layout: bool = True):
    mentions = scan_block(block, taxonomy, has_layout)
    seen = {m.skill_id for m in mentions}
    for _key, line_tokens in _line_groups(block.tokens):
        line = " ".join(t.text for t in line_tokens)
        for skill_id in taxonomy.skills_for_certification(line):
            if skill_id in seen or skill_id not in taxonomy.by_id:
                continue
            seen.add(skill_id)
            mentions.append(SkillMention(
                skill_id, taxonomy.by_id[skill_id]["skill_name"], _clean(line),
                "certification", line_tokens[0].page, _bbox(line_tokens, has_layout),
                "certification_map", line_tokens[0].start, line_tokens[-1].end))
    return mentions


def extract_skill_mentions(blocks, taxonomy: SkillTaxonomy, has_layout: bool = True):
    mentions: List[SkillMention] = []
    for block in blocks:
        if block.section_type == "certification":
            mentions.extend(scan_certifications(block, taxonomy, has_layout))
        elif block.section_type in ("skills_section", "experience", "project"):
            mentions.extend(scan_block(block, taxonomy, has_layout))
    return mentions


_CHUNK_SPLIT = re.compile(r"[,;|/]|\s{2,}|\u2022")
_STOPWORD_CHUNK = re.compile(
    r"^(technical skills?|programming languages?|skills?|tools?|others?|etc|and|none|"
    r"languages?|frameworks?|libraries|databases?|competencies|coursework|"
    r"skills learnt|core competencies|networking tools?|web tech)$", re.I)


def report_unresolved(blocks, taxonomy: SkillTaxonomy) -> Dict[str, int]:
    """Skills-section terms reaching no taxonomy entry. Mine this to grow coverage."""
    unresolved: Dict[str, int] = {}
    for block in blocks:
        if block.section_type != "skills_section":
            continue
        for line in block.text.split("\n"):
            body = line.split(":", 1)[-1] if ":" in line else line
            for chunk in _CHUNK_SPLIT.split(body):
                term = _clean(chunk)
                if not term or len(term) < 2 or _STOPWORD_CHUNK.match(term):
                    continue
                if taxonomy.lookup_surface(term) or taxonomy.link(term, original=term):
                    continue
                unresolved[term] = unresolved.get(term, 0) + 1
    return unresolved
