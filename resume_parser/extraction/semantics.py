"""Stage 4 - semantic filtering of skill mentions.

The gazetteer answers "does this term appear?". That is not the same question as
"does this candidate have this skill". Four ways they come apart:

    negation     "not production experience with Kubernetes"
    aspiration   "seeking to learn Rust", "currently learning Go"
    requirement  a pasted job description listing skills the candidate lacks
    object       "analysed Android malware"  <- Android is the subject matter

The first three are genuine rejections. The fourth is NOT, and treating it as
one was a real bug: in

    "Analysed Android malware samples using static analysis"

the term *Android* is the object being studied, so Android Development is wrong.
But the sentence is strong evidence of Malware Analysis. Discarding the whole
mention throws away the very capability the sentence demonstrates.

So object-of-analysis produces a REDIRECT, not a rejection: the object mention
is dropped and an activity mention is emitted in its place, derived from the
verb-object pair. Where no activity can be derived, the mention is dropped and
the reason is logged - never silently.

A missed skill is recoverable; a fabricated one is not. When a cue fires and no
redirect applies, this layer drops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

WINDOW_BEFORE = 60
WINDOW_AFTER = 30

NEGATION = re.compile(
    r"\b(no|not|non|never|without|lack(?:s|ing)?|absent|"
    r"unfamiliar|inexperienced|excluding|except)\b", re.I)

# Every cue needs a companion word. A bare "learning" would match the skill
# names "Machine Learning" and "Deep Learning" and silently suppress every
# skill listed after them.
ASPIRATION = re.compile(
    r"\b(seeking\s+to|aspir(?:e|ing)\s+to|want(?:ing)?\s+to\s+learn|"
    r"looking\s+to\s+learn|plan(?:ning)?\s+to\s+learn|"
    r"currently\s+learning|started\s+learning|began\s+learning|"
    r"interested\s+in\s+learning|keen\s+to\s+learn|"
    r"would\s+like\s+to\s+learn|hope\s+to\s+learn|eager\s+to\s+learn)\b", re.I)

REQUIREMENT = re.compile(
    r"\b(required|requirements?|must\s+have|should\s+have|we\s+are\s+(?:hiring|looking)|"
    r"preferred\s+qualifications?|job\s+description|responsibilities\s+include|"
    r"ideal\s+candidate|you\s+will\s+need)\b", re.I)

OBJECT_VERBS = re.compile(
    r"\b(analy[sz]ed|analy[sz]ing|analysis\s+of|studied|studying|research(?:ed|ing)?\s+on|"
    r"investigat(?:ed|ing)|review(?:ed|ing)|survey(?:ed|ing)|audit(?:ed|ing)|"
    r"detect(?:ed|ing|ion\s+of)|classif(?:ied|ying|ication\s+of)|"
    r"scann(?:ed|ing)|reverse[- ]engineer(?:ed|ing)?|dissect(?:ed|ing)?)\b", re.I)

# Nouns that, immediately after a technology name, turn it into subject matter.
OBJECT_TAILS = re.compile(
    r"^\s*(malware|virus|viruses|threats?|attacks?|vulnerabilit(?:y|ies)|"
    r"exploits?|samples?|binaries|apks?|datasets?|corpus|papers?|literature|"
    r"logs?|traffic|packets?)\b", re.I)

# verb-family, object-family -> the capability the sentence actually evidences.
# Keeps the evidence that "analysed X" demonstrates an analysis skill, while
# refusing to credit X itself as a skill.
ACTIVITY_REDIRECTS: List[Tuple[re.Pattern, re.Pattern, str]] = [
    (re.compile(r"analy[sz]|reverse[- ]engineer|dissect|detect|classif|scann", re.I),
     re.compile(r"malware|virus|apk|binar|exploit|payload", re.I),
     "malware analysis"),
    (re.compile(r"analy[sz]|monitor|inspect|captur", re.I),
     re.compile(r"traffic|packets?|logs?", re.I),
     "network security"),
    (re.compile(r"analy[sz]|process|model|explor", re.I),
     re.compile(r"datasets?|data\b|corpus", re.I),
     "data analysis"),
    (re.compile(r"detect|classif|identif", re.I),
     re.compile(r"threats?|attacks?|vulnerabilit|intrusion", re.I),
     "cybersecurity"),
]


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    reason: Optional[str] = None            # negation | aspiration | requirement | object
    redirect_surface: Optional[str] = None  # link this instead of the object


def _clause_before(text: str, start: int) -> str:
    """Text back to the nearest clause boundary, capped by WINDOW_BEFORE.

    Stopping at a boundary keeps "Python. Not familiar with Rust" from letting
    the negation reach back onto Python.
    """
    window = text[max(0, start - WINDOW_BEFORE):start]
    for sep in (";", ".", "•", "\n", " - ", "|"):
        idx = window.rfind(sep)
        if idx != -1:
            window = window[idx + len(sep):]
    return window


def _redirect_for(before: str, after: str) -> Optional[str]:
    for verb, obj, surface in ACTIVITY_REDIRECTS:
        if verb.search(before) and obj.search(after):
            return surface
    return None


def classify_mention(text: str, start: int, end: int,
                     section_type: str = "") -> Verdict:
    """Decide whether a matched span is a genuine claim of skill."""
    before = _clause_before(text, start)
    after = text[end:end + WINDOW_AFTER]

    if REQUIREMENT.search(before):
        return Verdict(False, "requirement")
    if NEGATION.search(before):
        return Verdict(False, "negation")
    if ASPIRATION.search(before):
        return Verdict(False, "aspiration")

    tail_is_object = bool(OBJECT_TAILS.match(after))
    verb_before = bool(OBJECT_VERBS.search(before))

    if tail_is_object and (verb_before or section_type in ("project", "experience")):
        # The term is subject matter, but the surrounding activity may still be
        # a real capability - carry it forward rather than losing the evidence.
        return Verdict(False, "object", _redirect_for(before, after))

    return Verdict(True)


def filter_mentions(mentions, text: str, taxonomy=None) -> Tuple[list, List[str]]:
    """Apply `classify_mention`, honouring activity redirects.

    Each mention needs `.char_start`, `.char_end`, `.source`, `.skill_name`.
    Redirects are linked through the taxonomy so the substituted skill still
    gets a real skill_id; if it cannot be linked, nothing is invented.
    """
    kept, warnings = [], []
    seen_redirects = set()

    for mention in mentions:
        start = getattr(mention, "char_start", None)
        if start is None:
            kept.append(mention)
            continue

        verdict = classify_mention(text, start, mention.char_end, mention.source)
        if verdict.accepted:
            kept.append(mention)
            continue

        redirected = False
        if verdict.redirect_surface and taxonomy is not None:
            match = taxonomy.link(verdict.redirect_surface)
            if match:
                key = (match.skill_id, mention.source)
                if key not in seen_redirects:
                    seen_redirects.add(key)
                    import dataclasses
                    kept.append(dataclasses.replace(
                        mention, skill_id=match.skill_id, skill_name=match.skill_name,
                        raw_text=mention.raw_text, match_rung="activity_redirect"))
                redirected = True
                warnings.append(
                    f"MENTION_REDIRECTED: '{mention.skill_name}' in {mention.source} "
                    f"is the object of analysis; recorded '{match.skill_name}' instead")

        if not redirected:
            warnings.append(
                f"MENTION_REJECTED: '{mention.skill_name}' in {mention.source} "
                f"({verdict.reason})")

    return kept, warnings
