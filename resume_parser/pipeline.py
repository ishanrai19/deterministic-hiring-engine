"""Resume Screening Agent - L0..L2. Document in, CandidateProfile out.

 0 ingest            PDF/CSV -> text + tokens + geometry
 1 segment           header dictionary, or LLM for non-standard headings
 2 locate entities   education, work history, project/role narratives
 3 extract mentions  gazetteer scan (not a learned NER model)
 4 semantic filter   reject negated/aspirational/JD terms; redirect objects
 5 link              exact -> alias -> acronym -> fuzzy -> embedding -> unresolved
 6 aggregate         merge by skill_id, evidence counts
 7 verify            re-read every bbox from the PDF
 8 validate          logical checks, parsing_status
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

from .evidence.aggregator import aggregate, verify_locations
from .extraction.entities import extract_education, extract_work_history
from .extraction.narratives import attach_role_descriptions, extract_projects
from .extraction.sections import apply_fallback, detect_sections
from .extraction.semantics import filter_mentions
from .extraction.skills import extract_skill_mentions, report_unresolved
from .ingestion.loaders import Document, load_csv, load_pdf
from .normalization.dates import experience_years, find_stated_experience
from .normalization.taxonomy import get_taxonomy
from .schemas.candidate_profile import SCHEMA_VERSION, CandidateProfile
from .validation.validators import logical_checks, resolve_status


@dataclass
class LLMConfig:
    enabled: bool = False
    model: str = "qwen2.5-32b-instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key: Optional[str] = None
    cache_dir: Optional[str] = None
    timeout: float = 60.0
    _client: object = None
    _cache: object = None

    def client_and_cache(self):
        if self._client is None:
            import os
            from .llm.segmenter import OpenAICompatibleClient, SegmentCache
            self._client = OpenAICompatibleClient(
                model=self.model, base_url=self.base_url,
                api_key=self.api_key or os.environ.get("RESUME_LLM_API_KEY", "-"),
                timeout=self.timeout)
            self._cache = SegmentCache(Path(self.cache_dir)) if self.cache_dir else None
        return self._client, self._cache


@dataclass
class ParseResult:
    profile: CandidateProfile
    warnings: List[str] = field(default_factory=list)
    unresolved: dict = field(default_factory=dict)
    segmenter: str = "deterministic"

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.profile.model_dump(mode="json", exclude_none=False),
                          indent=indent, ensure_ascii=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _deterministic(doc: Document):
    return apply_fallback(detect_sections(doc))


def _segment(doc: Document, llm: Optional[LLMConfig]):
    if not (llm and llm.enabled):
        blocks, warnings = _deterministic(doc)
        return blocks, warnings, "deterministic"
    from .llm.segmenter import segment_with_llm
    client, cache = llm.client_and_cache()
    result = segment_with_llm(doc, client, cache, llm.model)
    if result.blocks is not None:
        return result.blocks, result.warnings, "llm"
    blocks, warnings = _deterministic(doc)
    return blocks, warnings + result.warnings + [
        "LLM_FALLBACK: used deterministic section detection"], "deterministic"


def parse_document(doc: Document, today: Optional[date] = None,
                   llm: Optional[LLMConfig] = None, embeddings: bool = False,
                   semantic_filter: bool = True) -> ParseResult:
    today = today or date.today()
    warnings: List[str] = list(doc.errors)

    if not doc.text.strip():
        return ParseResult(
            CandidateProfile(candidate_id=doc.candidate_id,
                             source_resume_path=doc.source_path,
                             parsing_status="failed", parsing_timestamp=_now()),
            warnings + ["UNREADABLE: no text could be extracted"])

    taxonomy = get_taxonomy(embeddings)
    blocks, seg_warnings, segmenter = _segment(doc, llm)
    warnings.extend(seg_warnings)

    education, edu_w = extract_education(blocks)
    warnings.extend(edu_w)
    work_history, work_w = extract_work_history(blocks)
    warnings.extend(work_w)

    # Narratives: what the candidate actually built, in their own words. Kept
    # apart from `skills`, which is normalised and therefore unquotable.
    work_history = attach_role_descriptions(work_history, blocks)
    projects, project_w = extract_projects(blocks)
    warnings.extend(project_w)

    years, exp_w = experience_years(
        [(j.start_date, j.end_date) for j in work_history], today=today)

    # Employment intervals are authoritative. Only when there are none do we
    # fall back to an explicitly stated professional total.
    experience_from_text = False
    if years is None:
        stated = find_stated_experience(doc.text)
        if stated is not None:
            years, experience_from_text = stated, True
            warnings.append("EXPERIENCE_FROM_TEXT: experience_years read from a stated "
                            "total in the resume text, not from employment dates")
        else:
            warnings.extend(exp_w)
    else:
        warnings.extend(exp_w)

    mentions = extract_skill_mentions(blocks, taxonomy, has_layout=doc.has_layout)
    if semantic_filter:
        mentions, sem_warnings = filter_mentions(mentions, doc.text, taxonomy)
        warnings.extend(sem_warnings)

    skills = aggregate(mentions)
    if doc.has_layout and doc.source_path.lower().endswith(".pdf"):
        skills, span_w = verify_locations(skills, doc.source_path)
        warnings.extend(span_w)

    profile = CandidateProfile(
        candidate_id=doc.candidate_id, source_resume_path=doc.source_path,
        education=education, experience_years=years, work_history=work_history,
        projects=projects, skills=skills, parsing_status="partial",
        schema_version=SCHEMA_VERSION,
        parsing_timestamp=_now())

    warnings.extend(logical_checks(profile, today=today,
                                   experience_from_text=experience_from_text))
    profile.parsing_status = resolve_status(profile, warnings)
    return ParseResult(profile, warnings, report_unresolved(blocks, taxonomy), segmenter)


def parse_pdf(path, today=None, llm=None, embeddings=False, semantic_filter=True) -> ParseResult:
    return parse_document(load_pdf(path), today=today, llm=llm,
                          embeddings=embeddings, semantic_filter=semantic_filter)


def parse_csv(path, text_column=None, id_column=None, today=None, llm=None,
              limit=None, embeddings=False, semantic_filter=True) -> Iterable[ParseResult]:
    for doc in load_csv(path, text_column=text_column, id_column=id_column, limit=limit):
        yield parse_document(doc, today=today, llm=llm,
                             embeddings=embeddings, semantic_filter=semantic_filter)


def parse_path(path, text_column=None, id_column=None, today=None, llm=None,
               limit=None, embeddings=False, semantic_filter=True) -> Iterable[ParseResult]:
    path = Path(path)
    kw = dict(today=today, llm=llm, embeddings=embeddings, semantic_filter=semantic_filter)
    if path.is_dir():
        for pdf in sorted(path.rglob("*.pdf")):
            yield parse_pdf(pdf, **kw)
        for csv in sorted(path.rglob("*.csv")):
            yield from parse_csv(csv, text_column, id_column, limit=limit, **kw)
    elif path.suffix.lower() == ".pdf":
        yield parse_pdf(path, **kw)
    elif path.suffix.lower() in (".csv", ".tsv"):
        yield from parse_csv(path, text_column, id_column, limit=limit, **kw)
    else:
        raise ValueError(f"unsupported input: {path}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Resume Screening Agent (L0-L2)")
    ap.add_argument("input", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--text-column")
    ap.add_argument("--id-column")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--warnings", action="store_true")
    ap.add_argument("--unresolved", action="store_true")
    ap.add_argument("--taxonomy-info", action="store_true")
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--no-semantic-filter", action="store_true")
    g = ap.add_argument_group("LLM segmentation (optional)")
    g.add_argument("--llm", action="store_true")
    g.add_argument("--llm-model", default="qwen2.5-32b-instruct")
    g.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    g.add_argument("--llm-api-key", default=None)
    g.add_argument("--llm-cache", default=None)
    g.add_argument("--llm-timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    if args.taxonomy_info:
        print(json.dumps(get_taxonomy(args.embeddings).stats(), indent=2))
        return 0
    if not args.input:
        ap.error("input is required unless --taxonomy-info is used")

    llm = LLMConfig(enabled=args.llm, model=args.llm_model, base_url=args.llm_base_url,
                    api_key=args.llm_api_key, cache_dir=args.llm_cache,
                    timeout=args.llm_timeout) if args.llm else None
    results = list(parse_path(args.input, args.text_column, args.id_column, llm=llm,
                              limit=args.limit, embeddings=args.embeddings,
                              semantic_filter=not args.no_semantic_filter))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for r in results:
                fh.write(r.to_json(indent=None) + "\n")
        print(f"wrote {len(results)} profile(s) to {out}", file=sys.stderr)
    else:
        for r in results:
            print(r.to_json(indent=2 if args.pretty else None))

    if args.warnings:
        for r in results:
            for w in r.warnings:
                print(f"[{r.profile.candidate_id}] {w}", file=sys.stderr)

    if args.unresolved:
        merged: dict = {}
        for r in results:
            for term, count in r.unresolved.items():
                merged[term] = merged.get(term, 0) + count
        print("\n-- unresolved terms (add to overlay_skills.json) --", file=sys.stderr)
        for term, count in sorted(merged.items(), key=lambda kv: -kv[1])[:40]:
            print(f"  {count:5d}  {term}", file=sys.stderr)

    statuses: dict = {}
    segmenters: dict = {}
    for r in results:
        statuses[r.profile.parsing_status] = statuses.get(r.profile.parsing_status, 0) + 1
        segmenters[r.segmenter] = segmenters.get(r.segmenter, 0) + 1
    print(f"status summary: {statuses}", file=sys.stderr)
    print(f"segmenter usage: {segmenters}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
