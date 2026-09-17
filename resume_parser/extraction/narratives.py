"""Project and role narratives - title plus the candidate's own description.

Why this is separate from skills
--------------------------------
`skills` answers "what can this person do". It is normalised, deduplicated and
linked to a taxonomy, which is exactly what the Matching Agent needs and exactly
what makes it useless for conversation: "Python, Docker, PyTorch" gives an
interviewer nothing specific to ask about.

`projects` and the per-role `description` answer "what did they actually build".
They are kept verbatim so the Interview Agent can quote them back, and so a
recruiter can see the claim in the candidate's own words.

Nothing here is paraphrased, summarised by a model, or invented. Text that is
not in the resume never appears in the output - only truncation is applied, and
only at a word boundary with an ellipsis so the cut is visible.

Entry detection
---------------
Within a project or experience block, a line is a TITLE when it reads like a
heading rather than a sentence: short, not starting with an action verb, not a
sub-bullet. Everything up to the next title becomes that entry's description.
That single heuristic covers the common resume shapes:

    Banking Web Application (TransactiWar)      <- title
    - Developed a secure banking app using PHP  <- description
    - Implemented session management            <- description

    Uncertainty Estimation in LLM | Guide: ...  <- title
    M.Tech Thesis - Ongoing                     <- metadata, folded in
    Developing a novel method to estimate ...   <- description
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ..normalization.dates import find_date_range
from ..schemas.candidate_profile import Project, WorkHistory
from .sections import Block

# Descriptions are evidence, not prose to read end-to-end. Long enough to be
# specific, short enough that the downstream payload stays compact.
MAX_DESCRIPTION_CHARS = 320
MAX_TITLE_WORDS = 14

_BULLET = re.compile(r"^\s*[-•*\u2022\u25cf\u25aa>]\s*")

# A line opening with an action verb is describing work, not naming it.
# Both past and present-participle forms: resumes mix "Built a pipeline" with
# "Developing a novel method", and only catching the former turned description
# lines into titles.
_ACTION_STEMS = (
    "build|built|design|develop|implement|create|lead|led|manage|engineer|"
    "architect|deploy|automate|automat|optimi[sz]e|optimi[sz]|improve|improv|"
    "integrate|integrat|migrate|migrat|analy[sz]e|analy[sz]|research|train|"
    "test|maintain|refactor|collaborate|collaborat|work|use|using|apply|"
    "appli|extend|devise|devis|conduct|perform|deliver|reduce|reduc|increase|"
    "increas|achieve|achiev|profile|profil|investigate|investigat|explore|"
    "explor|prototype|prototyp|write|wrote|scale|scal|configure|configur|"
    "handle|handl|support|study|studying|studied"
)
_ACTION_VERB = re.compile(rf"^(?:{_ACTION_STEMS})(?:e?d|ing|s)?\b", re.I)

# Trailing decoration on a title line: "| PyTorch, CUB-200 (GitHub)".
_TITLE_TAIL = re.compile(r"\s*[|(\[].*$")

# Lines that are metadata about an entry rather than its description.
_METADATA = re.compile(
    r"^\s*("
    r"(?:m\.?tech|b\.?tech|course|capstone|semester|final\s+year)\s+"
    r"(?:thesis|project|work)"
    r"|tech(?:nology)?\s+stack\s*:"
    r"|technologies\s+used\s*:"
    r"|tools?\s+used\s*:"
    r"|guide\s*:"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[\s\-–—]*\d{4}"
    r"|\d{4}\s*[-–—]\s*(?:\d{4}|present|ongoing)"
    r")", re.I)


def _is_title(line: str, *, previous_was_title: bool) -> bool:
    """A heading names something; a description says what was done to it."""
    stripped = line.strip()
    if not stripped:
        return False
    if _BULLET.match(stripped):
        return False                      # sub-bullets are always description
    if _ACTION_VERB.match(stripped):
        return False
    if _METADATA.match(stripped):
        return False
    if len(stripped.split()) > MAX_TITLE_WORDS:
        return False
    if stripped.endswith((".", ";")):
        return False                      # a sentence, not a heading
    if previous_was_title:
        return False                      # two headings in a row: the second is a subtitle
    return True


def _clean_line(line: str) -> str:
    return _BULLET.sub("", line).strip()


def _truncate(text: str) -> Optional[str]:
    """Cut at a word boundary and mark the cut, so nothing looks complete
    when it is not."""
    text = re.sub(r"\s+", " ", text).strip(" ;,-–—")
    if not text:
        return None
    if len(text) <= MAX_DESCRIPTION_CHARS:
        return text
    cut = text[:MAX_DESCRIPTION_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return cut + "…"


def _split_entries(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """Group lines into (title, body-lines) pairs."""
    entries: List[Tuple[str, List[str]]] = []
    previous_was_title = False
    for line in lines:
        if _is_title(line, previous_was_title=previous_was_title):
            entries.append((line.strip(), []))
            previous_was_title = True
        else:
            if entries:
                entries[-1][1].append(line)
            previous_was_title = False
    return entries


def _entry_dates(line: str) -> Tuple[Optional[str], Optional[str]]:
    """Date range for an entry.

    A lone "Ongoing"/"Present" marks the end, not the start - reading it as a
    start date would claim the work began today.
    """
    start, end = find_date_range(line)
    if start == "Present" and end is None:
        return None, "Present"
    return start, end


def _describe(body: List[str]) -> Optional[str]:
    """Join body lines, dropping pure metadata but keeping tech-stack notes."""
    parts: List[str] = []
    for line in body:
        cleaned = _clean_line(line)
        if not cleaned:
            continue
        if _METADATA.match(cleaned) and not re.search(r":", cleaned):
            continue                      # bare dates and thesis labels
        parts.append(cleaned)
    return _truncate(" ".join(parts))


def extract_projects(blocks: List[Block]) -> Tuple[List[Project], List[str]]:
    """Titles and descriptions from project blocks, in document order."""
    projects: List[Project] = []
    warnings: List[str] = []
    seen: set = set()

    for block in blocks:
        if block.section_type != "project":
            continue
        lines = [l for l in block.text.split("\n") if l.strip()]
        for title, body in _split_entries(lines):
            clean_title = _TITLE_TAIL.sub("", title).strip(" -–—|")
            if not clean_title or len(clean_title) < 3:
                continue
            key = clean_title.lower()
            if key in seen:
                continue
            seen.add(key)

            start, end = _entry_dates(title)
            if start is None and end is None:
                for line in body[:2]:
                    start, end = _entry_dates(line)
                    if start or end:
                        break

            projects.append(Project(title=clean_title, description=_describe(body),
                                    start_date=start, end_date=end))

    if not projects:
        warnings.append("PROJECTS_EMPTY: no project entries were recognised")
    return projects, warnings


def attach_role_descriptions(work_history: List[WorkHistory],
                             blocks: List[Block]) -> List[WorkHistory]:
    """Fill `description` on each role from the bullets beneath its heading.

    Matching is by company or title appearing in the entry heading, so a role
    only ever receives its own bullets. Unmatched roles keep description=None
    rather than borrowing text from a neighbour.
    """
    if not work_history:
        return work_history

    entries: List[Tuple[str, List[str]]] = []
    for block in blocks:
        if block.section_type == "experience":
            entries.extend(_split_entries([l for l in block.text.split("\n") if l.strip()]))

    for role in work_history:
        needles = [v.lower() for v in (role.company, role.title) if v]
        if not needles:
            continue
        for title, body in entries:
            haystack = title.lower()
            if any(n in haystack for n in needles):
                role.description = _describe(body)
                break
    return work_history
