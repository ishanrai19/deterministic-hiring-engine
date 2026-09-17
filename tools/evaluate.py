"""Score parser output against ground truth.

    python tools/evaluate.py --gold gold/                       labelled resumes
    python tools/evaluate.py --csv r.csv --profiles out.jsonl   labelled columns

Gold mode reports precision as well as recall: on real resumes the expensive
error is a skill the parser invented, not one it missed. Only files marked
`reviewed: true` are scored - an unreviewed draft is parser output, and scoring
against it would measure the parser against itself.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resume_parser.pipeline import parse_pdf

GOLD_SUFFIX = ".expected.json"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", (text or "").lower())


def _skill_set(values) -> Set[str]:
    return {_norm(v) for v in values if _norm(v)}


def _predicted_surfaces(profile) -> Set[str]:
    out = {_norm(s.skill_name) for s in profile.skills}
    out |= {_norm(loc.raw_text or "") for s in profile.skills for loc in s.locations}
    out.discard("")
    return out


def evaluate_gold(folder: Path, show: int, embeddings: bool, semantic_filter: bool) -> int:
    pairs, unreviewed = [], 0
    for pdf in sorted(folder.rglob("*.pdf")):
        gold = pdf.with_suffix("").with_name(pdf.stem + GOLD_SUFFIX)
        if not gold.exists():
            continue
        payload = json.loads(gold.read_text(encoding="utf-8"))
        if payload.get("reviewed"):
            pairs.append((pdf, payload))
        else:
            unreviewed += 1

    if not pairs:
        print(f"no reviewed gold files in {folder}"
              + (f" ({unreviewed} draft(s) awaiting review)" if unreviewed else ""))
        print("run tools/make_gold.py, correct the drafts, then set reviewed=true")
        return 1

    tp = fp = fn = 0
    miss, spurious = collections.Counter(), collections.Counter()
    exp_exact = exp_wrong = exp_null = exp_nulltruth = 0
    edu_exact = edu_partial = edu_wrong = 0
    statuses: collections.Counter = collections.Counter()
    per_file = []

    for pdf, gold in pairs:
        profile = parse_pdf(str(pdf), embeddings=embeddings,
                            semantic_filter=semantic_filter).profile
        statuses[profile.parsing_status] += 1

        truth = _skill_set(gold.get("skills", []))
        predicted_names = {_norm(s.skill_name) for s in profile.skills}
        predicted_all = _predicted_surfaces(profile)

        hits = {t for t in truth if t in predicted_all}
        missed = truth - hits
        extra = {n for n in predicted_names if n not in truth}
        tp += len(hits); fn += len(missed); fp += len(extra)
        for t in missed:
            miss[t] += 1
        for t in extra:
            spurious[t] += 1

        truth_years, got_years = gold.get("experience_years"), profile.experience_years
        if truth_years is None:
            exp_nulltruth += 1
            if got_years is not None:
                exp_wrong += 1
        elif got_years is None:
            exp_null += 1
        elif abs(float(got_years) - float(truth_years)) < 0.01:
            exp_exact += 1
        else:
            exp_wrong += 1

        truth_deg = {(_norm(e.get("degree") or ""), _norm(e.get("institution") or ""))
                     for e in gold.get("education", [])}
        got_deg = {(_norm(e.degree or ""), _norm(e.institution or ""))
                   for e in profile.education}
        if truth_deg == got_deg:
            edu_exact += 1
        elif truth_deg & got_deg:
            edu_partial += 1
        else:
            edu_wrong += 1

        per_file.append((pdf.name, len(hits), len(missed), len(extra),
                         profile.parsing_status))

    print(f"reviewed resumes: {len(pairs)}"
          + (f"   (skipped {unreviewed} unreviewed)" if unreviewed else ""))
    print(f"parsing status  : {dict(statuses)}")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"\nSKILLS\n  precision : {tp}/{tp + fp} = {precision:.2%}"
          f"   (invented skills cost the most)")
    print(f"  recall    : {tp}/{tp + fn} = {recall:.2%}")
    print(f"  F1        : {f1:.2%}")
    if miss:
        print(f"  top missed ({show}) - taxonomy gaps:")
        for term, count in miss.most_common(show):
            print(f"    {count:4d}  {term}")
    if spurious:
        print(f"  top spurious ({show}) - precision bugs:")
        for term, count in spurious.most_common(show):
            print(f"    {count:4d}  {term}")

    denom = exp_exact + exp_wrong + exp_null
    print("\nEXPERIENCE_YEARS")
    if denom:
        print(f"  exact : {exp_exact}/{denom} = {exp_exact / denom:.2%}")
        print(f"  wrong : {exp_wrong}\n  null when truth known : {exp_null}")
    print(f"  correctly null (no basis in resume) : {exp_nulltruth}")

    total = len(pairs)
    print(f"\nEDUCATION (degree + institution sets)")
    print(f"  exact {edu_exact}/{total} · partial {edu_partial}/{total} · none {edu_wrong}/{total}")

    print("\nPER FILE  (hit / miss / spurious / status)")
    for name, hit, missed_n, extra_n, status in per_file:
        flag = "  <-- review" if extra_n or missed_n > 2 else ""
        print(f"  {name[:44]:46} {hit:3} {missed_n:3} {extra_n:3}  {status}{flag}")
    return 0


def evaluate_csv(csv_path: Path, profiles_path: Path, id_col: str, skills_col: str,
                 exp_col: str, show: int) -> int:
    import pandas as pd

    frame = pd.read_csv(csv_path, dtype=str).fillna("")
    profiles: Dict[str, dict] = {}
    for line in profiles_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            profiles[payload["candidate_id"]] = payload

    matched = missed = extra = unmatched = 0
    miss_counter: collections.Counter = collections.Counter()
    exp_exact = exp_wrong = exp_null = 0
    statuses: collections.Counter = collections.Counter()

    for _, row in frame.iterrows():
        profile = profiles.get(str(row.get(id_col, "")).strip())
        if profile is None:
            unmatched += 1
            continue
        statuses[profile["parsing_status"]] += 1
        if skills_col in frame.columns:
            truth = {_norm(p) for p in re.split(r"[,;|]", row[skills_col]) if p.strip()}
            got = {_norm(s["skill_name"]) for s in profile.get("skills", [])}
            got |= {_norm(l.get("raw_text") or "")
                    for s in profile.get("skills", []) for l in s.get("locations", [])}
            for term in truth:
                if term in got:
                    matched += 1
                else:
                    missed += 1
                    miss_counter[term] += 1
            extra += sum(1 for s in profile.get("skills", [])
                         if _norm(s["skill_name"]) not in truth)
        if exp_col in frame.columns:
            try:
                truth_years = float(str(row[exp_col]).strip())
            except ValueError:
                continue
            got_years = profile.get("experience_years")
            if got_years is None:
                exp_null += 1
            elif abs(got_years - truth_years) < 0.01:
                exp_exact += 1
            else:
                exp_wrong += 1

    total = len(frame) - unmatched
    print(f"rows evaluated: {total}" + (f"  (unmatched ids: {unmatched})" if unmatched else ""))
    print(f"parsing status: {dict(statuses)}")
    if matched + missed:
        print(f"\nSKILLS\n  recall : {matched}/{matched + missed} = "
              f"{matched / (matched + missed):.2%}\n  extra  : {extra}")
        for term, count in miss_counter.most_common(show):
            print(f"    {count:5d}  {term}")
    denom = exp_exact + exp_wrong + exp_null
    if denom:
        print(f"\nEXPERIENCE_YEARS\n  exact : {exp_exact}/{denom} = {exp_exact / denom:.2%}")
        print(f"  wrong : {exp_wrong}\n  null  : {exp_null}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate parser output")
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--profiles", type=Path)
    ap.add_argument("--id-column", default="resume_id")
    ap.add_argument("--skills-column", default="resume_skills")
    ap.add_argument("--experience-column", default="experience_years")
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--no-semantic-filter", action="store_true")
    args = ap.parse_args(argv)

    if args.gold:
        return evaluate_gold(args.gold, args.show, args.embeddings,
                             not args.no_semantic_filter)
    if args.csv and args.profiles:
        return evaluate_csv(args.csv, args.profiles, args.id_column,
                            args.skills_column, args.experience_column, args.show)
    ap.error("pass --gold FOLDER, or --csv FILE with --profiles FILE")


if __name__ == "__main__":
    raise SystemExit(main())
