"""Generate editable gold-standard drafts from parser output.

Writing resume JSONs by hand is miserable. Instead the parser proposes and you
correct: this writes `<resume>.expected.json` beside each PDF, pre-filled with
current parser output plus the raw text so you can check without opening the PDF.

    python tools/make_gold.py gold/                 create drafts
    python tools/make_gold.py gold/ --status        labelling progress
    python tools/make_gold.py gold/ --refresh       re-propose ONLY unreviewed files

Correct four fields, then set "reviewed": true:

    education        one entry per real qualification
    experience_years a number, or null if the resume gives no basis
    work_history     employers and roles only, each with its own description
    projects         title + the candidate's own description, verbatim
    skills           the canonical names a recruiter would agree with

Only reviewed files are scored, so a half-labelled folder still gives honest
numbers. Keys starting with _ are ignored by the scorer.

CAUTION: drafts are parser output. Skim-approving them means measuring the
parser against itself. Read the resume, not the draft.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resume_parser.pipeline import parse_pdf

GOLD_SUFFIX = ".expected.json"


def gold_path(pdf: Path) -> Path:
    return pdf.with_suffix("").with_name(pdf.stem + GOLD_SUFFIX)


def _load_text(pdf: Path) -> str:
    import fitz
    with fitz.open(pdf) as doc:
        return "\n".join(page.get_text() for page in doc)


def build_draft(pdf: Path) -> dict:
    result = parse_pdf(str(pdf))
    profile = result.profile
    return {
        "_instructions": "Correct the four fields below, then set reviewed=true. "
                         "Keys starting with _ are ignored by the scorer.",
        "reviewed": False,
        "source_pdf": pdf.name,
        "education": [e.model_dump(exclude_none=False) for e in profile.education],
        "experience_years": profile.experience_years,
        "work_history": [w.model_dump(exclude_none=False) for w in profile.work_history],
        "projects": [p.model_dump(exclude_none=False) for p in profile.projects],
        "skills": sorted(s.skill_name for s in profile.skills),
        "_parser_status": profile.parsing_status,
        "_parser_warnings": result.warnings,
        "_unresolved_terms": sorted(result.unresolved),
        "_raw_text": _load_text(pdf),
    }


def is_reviewed(path: Path) -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("reviewed"))
    except (json.JSONDecodeError, OSError):
        return False


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Create or refresh gold-standard drafts")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--refresh", action="store_true",
                    help="re-propose unreviewed drafts (reviewed files are never touched)")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args(argv)

    pdfs = sorted(args.folder.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs found in {args.folder}")

    if args.status:
        drafted = [p for p in pdfs if gold_path(p).exists()]
        reviewed = [p for p in drafted if is_reviewed(gold_path(p))]
        print(f"resumes        : {len(pdfs)}")
        print(f"drafts created : {len(drafted)}")
        print(f"reviewed       : {len(reviewed)}")
        pending = [p.name for p in drafted if not is_reviewed(gold_path(p))]
        if pending:
            print(f"awaiting review: {', '.join(pending[:10])}"
                  + (" ..." if len(pending) > 10 else ""))
        return 0

    created = refreshed = skipped = 0
    for pdf in pdfs:
        target = gold_path(pdf)
        if target.exists():
            if is_reviewed(target) or not args.refresh:
                skipped += 1
                continue
            refreshed += 1
        else:
            created += 1
        target.write_text(json.dumps(build_draft(pdf), indent=2, ensure_ascii=False),
                          encoding="utf-8")

    print(f"created {created}, refreshed {refreshed}, left alone {skipped}")
    print(f"\nNext: edit the .expected.json files in {args.folder}, set reviewed=true,")
    print(f"then run: python tools/evaluate.py --gold {args.folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
