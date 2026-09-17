"""LLM section segmenter (L1).

Scope discipline
----------------
The model is restricted to ONE decision: which of six labels applies to each
line range. It returns line numbers, never text, names, dates or IDs.

An LLM is not a deterministic component - inference and tooling can vary across
environments. The defensible claim is narrower and stronger: its output is
constrained to line ranges, schema-validated, cached by content hash, and never
used for scoring or ranking. The CandidateProfile stays reproducible because
every value in it is computed by deterministic code downstream.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple

from ..extraction.sections import ALL_SECTION_TYPES, Block, blocks_from_spans
from ..ingestion.loaders import Document

DEFAULT_MODEL = os.environ.get("RESUME_LLM_MODEL", "qwen2.5-32b-instruct")
DEFAULT_BASE_URL = os.environ.get("RESUME_LLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_TIMEOUT = float(os.environ.get("RESUME_LLM_TIMEOUT", "60"))
MAX_LINES = 400
MAX_LINE_CHARS = 200

SYSTEM_PROMPT = (
    "You label the sections of a resume. You are given the resume as numbered "
    "lines. Return JSON describing which contiguous line ranges belong to which "
    "section.\n\n"
    "Allowed section values, and nothing else:\n"
    "  skills_section - lists of skills, tools, technologies, competencies\n"
    "  experience     - paid employment, jobs, roles, internships\n"
    "  project        - projects, research, case studies\n"
    "  certification  - certifications, licences, registrations, credentials\n"
    "  education      - degrees, institutions, academic qualifications\n"
    "  other          - anything else: summary, awards, courses taken, hobbies,\n"
    "                   positions of responsibility, achievements, contact details\n\n"
    "Rules:\n"
    "1. Judge by meaning, not exact wording. 'Jobs Done', 'Where I Worked' and "
    "'Professional Background' are all 'experience'.\n"
    "2. A list of COURSES TAKEN or subjects studied is 'other', never "
    "'skills_section' and never 'project'.\n"
    "3. 'Positions of Responsibility' is 'other' unless clearly paid employment.\n"
    "4. Cover every line from first to last. Ranges must not overlap.\n"
    "5. Use 'other' when unsure. Never invent a section that is not present.\n"
    "6. Return only JSON. No prose, no markdown fences.")

USER_TEMPLATE = (
    "Resume lines:\n\n{numbered}\n\n"
    'Return: {{"sections": [{{"section": "<label>", "start_line": <int>, '
    '"end_line": <int>}}]}}\n'
    "Line numbers are inclusive and refer to the numbers shown above.")

SEGMENT_SCHEMA: Dict = {
    "type": "object",
    "properties": {"sections": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "section": {"type": "string", "enum": sorted(ALL_SECTION_TYPES)},
            "start_line": {"type": "integer", "minimum": 0},
            "end_line": {"type": "integer", "minimum": 0}},
        "required": ["section", "start_line", "end_line"],
        "additionalProperties": False}}},
    "required": ["sections"], "additionalProperties": False}


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass
class OpenAICompatibleClient:
    """vLLM, Ollama, Groq, OpenRouter - anything speaking /v1/chat/completions."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    api_key: str = field(default_factory=lambda: os.environ.get("RESUME_LLM_API_KEY", "-"))
    timeout: float = DEFAULT_TIMEOUT
    temperature: float = 0.0
    guided: bool = True

    def complete(self, system: str, user: str) -> str:
        from openai import OpenAI
        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        attempts = []
        if self.guided:
            attempts.append({"extra_body": {"structured_outputs": {"json": SEGMENT_SCHEMA}}})
            attempts.append({"response_format": {"type": "json_schema", "json_schema": {
                "name": "sections", "schema": SEGMENT_SCHEMA, "strict": True}}})
            attempts.append({"response_format": {"type": "json_object"}})
        attempts.append({})
        last: Optional[Exception] = None
        for kwargs in attempts:
            try:
                r = client.chat.completions.create(model=self.model, messages=messages,
                                                   temperature=self.temperature, **kwargs)
                return r.choices[0].message.content or ""
            except Exception as exc:
                last = exc
        raise RuntimeError(f"all LLM completion attempts failed: {last}")


def _line_index(doc: Document) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    cursor = 0
    for line in doc.text.split("\n"):
        out.append((cursor, cursor + len(line), line))
        cursor += len(line) + 1
    return out


def build_prompt(lines) -> str:
    numbered = "\n".join(f"{i:>3}| {t[:MAX_LINE_CHARS]}"
                         for i, (_s, _e, t) in enumerate(lines[:MAX_LINES]))
    return USER_TEMPLATE.format(numbered=numbered)


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.M)


def parse_response(raw: str) -> List[dict]:
    text = _FENCE.sub("", raw or "").strip()
    if not text:
        raise ValueError("empty LLM response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError("no JSON object in LLM response")
        payload = json.loads(match.group())
    sections = payload.get("sections")
    if not isinstance(sections, list) or not sections:
        raise ValueError("LLM response has no 'sections' array")
    return sections


def validate_sections(sections: List[dict], line_count: int) -> List[Tuple[str, int, int]]:
    """Reject anything the model got wrong; never repair it silently."""
    cleaned: List[Tuple[str, int, int]] = []
    for item in sections:
        if not isinstance(item, dict):
            raise ValueError("section entry is not an object")
        label, start, end = item.get("section"), item.get("start_line"), item.get("end_line")
        if label not in ALL_SECTION_TYPES:
            raise ValueError(f"unknown section label: {label!r}")
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("line numbers must be integers")
        if start < 0 or end < start:
            raise ValueError(f"invalid range: {start}..{end}")
        if start >= line_count:
            raise ValueError(f"start_line {start} beyond document ({line_count} lines)")
        cleaned.append((label, start, min(end, line_count - 1)))
    cleaned.sort(key=lambda t: t[1])
    for (_a, _b, e1), (_c, s2, _d) in zip(cleaned, cleaned[1:]):
        if s2 <= e1:
            raise ValueError(f"overlapping ranges at line {s2}")
    return cleaned


def spans_from_sections(sections, lines) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    for label, start_line, end_line in sections:
        for idx in range(start_line, min(end_line, len(lines) - 1) + 1):
            start, end, text = lines[idx]
            if text.strip():
                spans.append((label, start, end))
    return spans


@dataclass
class SegmentCache:
    """Content-hash cache: the same resume is not re-interpreted on every run."""

    directory: Optional[Path] = None

    def _path(self, content_hash: str, model: str) -> Optional[Path]:
        if self.directory is None:
            return None
        return Path(self.directory) / f"{content_hash}.{re.sub(r'[^w.-]+', '_', model)}.json"

    def get(self, content_hash: str, model: str) -> Optional[str]:
        p = self._path(content_hash, model)
        return p.read_text(encoding="utf-8") if p and p.exists() else None

    def put(self, content_hash: str, model: str, raw: str) -> None:
        p = self._path(content_hash, model)
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(raw, encoding="utf-8")


@dataclass
class SegmentationResult:
    blocks: Optional[List[Block]]
    warnings: List[str] = field(default_factory=list)
    used_llm: bool = False


def segment_with_llm(doc: Document, client: LLMClient, cache: Optional[SegmentCache] = None,
                     model_name: str = DEFAULT_MODEL) -> SegmentationResult:
    lines = _line_index(doc)
    if not lines:
        return SegmentationResult(None, ["LLM_SKIPPED: document has no lines"])
    raw = cache.get(doc.content_hash, model_name) if cache else None
    cached = raw is not None
    if raw is None:
        try:
            raw = client.complete(SYSTEM_PROMPT, build_prompt(lines))
        except Exception as exc:
            return SegmentationResult(None, [f"LLM_UNAVAILABLE: {exc}"])
    try:
        sections = validate_sections(parse_response(raw), min(len(lines), MAX_LINES))
    except Exception as exc:
        return SegmentationResult(None, [f"LLM_INVALID_RESPONSE: {exc}"])
    if cache and not cached:
        cache.put(doc.content_hash, model_name, raw)
    spans = spans_from_sections(sections, lines)
    if not spans:
        return SegmentationResult(None, ["LLM_EMPTY_SEGMENTATION: no usable spans"])
    blocks = blocks_from_spans(doc, spans)
    if not blocks:
        return SegmentationResult(None, ["LLM_EMPTY_SEGMENTATION: no blocks after mapping"])
    return SegmentationResult(blocks, ["LLM_SEGMENTATION_CACHED"] if cached else [], True)
