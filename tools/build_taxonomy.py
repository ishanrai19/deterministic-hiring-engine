"""Compile the runtime skill taxonomy from ESCO + the local overlay.

ESCO is the canonical vocabulary; the overlay is the extension layer for
technologies, brands and acronyms ESCO covers thinly.

Download "ESCO dataset - classification - en - csv" from
https://esco.ec.europa.eu/en/use-esco/download (free, 28 languages).

    python tools/build_taxonomy.py --esco-skills skills_en.csv \
        --hierarchy skillsHierarchy_en.csv \
        --out resume_parser/data/taxonomy_compiled.json --collisions

Run with --collisions and read the output: with ~15k concepts some surface forms
are claimed twice, and the runtime refuses to link those rather than guessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

_SPLIT_LABELS = re.compile(r"[\r\n]+")
MAX_SURFACE_WORDS = 6

# ESCO alt labels include bare words like "care" and "plan"; indexed naively
# these match every resume.
STOP_SURFACE: Set[str] = {
    "a","about","act","action","activity","advise","advice","aid","an","and","apply",
    "approach","area","assist","be","build","care","case","change","check","collect",
    "company","control","create","data","day","deal","design","develop","do","draw",
    "end","ensure","family","field","file","follow","form","get","give","goal","group",
    "guide","handle","have","help","hold","home","identify","in","issue","item","job",
    "keep","know","lead","level","life","line","list","maintain","make","manage","mark",
    "match","method","monitor","need","note","of","offer","on","operate","order",
    "organise","organize","part","pay","people","perform","place","plan","play","point",
    "prepare","present","process","product","provide","read","record","report","result",
    "review","run","safety","sell","service","set","show","site","skill","staff","start",
    "state","step","stock","study","support","system","take","task","team","test","the",
    "time","to","train","type","understand","unit","use","value","view","waste","way",
    "with","work","write",
}


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^\w#+./\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .-")


def _usable_surface(label: str) -> bool:
    key = _norm(label)
    if not key or len(key) < 2 or len(key.split()) > MAX_SURFACE_WORDS:
        return False
    return key not in STOP_SURFACE


def load_hierarchy(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    mapping: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        uri_cols = [c for c in fields if "uri" in c.lower()]
        label_cols = [c for c in fields if "preferredlabel" in c.lower().replace(" ", "")]
        if not uri_cols or not label_cols:
            print(f"warning: unrecognised hierarchy columns in {path}", file=sys.stderr)
            return {}
        leaf, top = uri_cols[-1], label_cols[0]
        for row in reader:
            uri, label = (row.get(leaf) or "").strip(), (row.get(top) or "").strip()
            if uri and label:
                mapping[uri] = label
    return mapping


def load_esco(skills_csv: Path, hierarchy, domains=None, skill_types=None) -> List[dict]:
    entries: List[dict] = []
    domain_filters = [d.lower() for d in (domains or [])]
    wanted = {t.lower() for t in (skill_types or [])}
    with open(skills_csv, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        f = {c.lower(): c for c in (reader.fieldnames or [])}
        c_uri, c_pref = f.get("concepturi"), f.get("preferredlabel")
        c_alt, c_hidden = f.get("altlabels"), f.get("hiddenlabels")
        c_type, c_status = f.get("skilltype"), f.get("status")
        if not c_uri or not c_pref:
            raise SystemExit(f"{skills_csv} is not ESCO skills_en.csv "
                             f"(need conceptUri + preferredLabel; saw {reader.fieldnames})")
        for row in reader:
            if c_status and (row.get(c_status) or "").strip().lower() == "obsolete":
                continue
            if wanted and c_type and (row.get(c_type) or "").strip().lower() not in wanted:
                continue
            uri, preferred = (row.get(c_uri) or "").strip(), (row.get(c_pref) or "").strip()
            if not uri or not preferred:
                continue
            if domain_filters and not any(d in hierarchy.get(uri, "").lower()
                                          for d in domain_filters):
                continue
            aliases: Set[str] = set()
            for col in (c_alt, c_hidden):
                if not col:
                    continue
                for label in _SPLIT_LABELS.split(row.get(col) or ""):
                    label = label.strip()
                    if label and _usable_surface(label):
                        aliases.add(label)
            # "care" is a legitimate concept name but a terrible search key.
            name_searchable = _usable_surface(preferred)
            if not name_searchable and not aliases:
                continue
            entries.append({"skill_id": uri, "skill_name": preferred,
                            "aliases": sorted(aliases), "acronyms": [],
                            "name_searchable": name_searchable,
                            "source": "esco", "group": hierarchy.get(uri, "")})
    return entries


def load_overlay(path: Optional[Path]) -> List[dict]:
    if not path or not Path(path).exists():
        return []
    out = []
    for entry in json.loads(Path(path).read_text(encoding="utf-8")).get("skills", []):
        entry.setdefault("aliases", [])
        entry.setdefault("acronyms", [])
        entry["source"] = "overlay"
        out.append(entry)
    return out


def load_overlay_cert_map(path: Optional[Path]) -> dict:
    if not path or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8")).get("certification_skill_map", {})


def merge(esco: Iterable[dict], overlay: Iterable[dict]) -> List[dict]:
    """Both are kept. The runtime applies overlay precedence per surface form,
    so an overlay entry overrides ESCO without erasing the ESCO concept."""
    return list(esco) + list(overlay)


def report_collisions(entries: List[dict]) -> Dict[str, List[str]]:
    owners: Dict[str, List[str]] = defaultdict(list)
    for entry in entries:
        if entry.get("source") == "overlay":
            continue        # overlay precedence is deliberate, not a collision
        labels = [*entry["aliases"], *entry["acronyms"]]
        if entry.get("name_searchable", True):
            labels.append(entry["skill_name"])
        for label in labels:
            key = _norm(label)
            if key and entry["skill_id"] not in owners[key]:
                owners[key].append(entry["skill_id"])
    return {k: v for k, v in owners.items() if len(v) > 1}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compile ESCO + overlay into a runtime taxonomy")
    ap.add_argument("--esco-skills", type=Path)
    ap.add_argument("--hierarchy", type=Path)
    ap.add_argument("--overlay", type=Path,
                    default=Path("resume_parser/data/overlay_skills.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("resume_parser/data/taxonomy_compiled.json"))
    ap.add_argument("--domain", action="append",
                    help="keep only ESCO skills whose group matches (repeatable)")
    ap.add_argument("--skill-type", action="append")
    ap.add_argument("--collisions", action="store_true")
    args = ap.parse_args(argv)

    hierarchy = load_hierarchy(args.hierarchy)
    esco = load_esco(args.esco_skills, hierarchy, args.domain, args.skill_type) \
        if args.esco_skills else []
    overlay = load_overlay(args.overlay)
    if not esco and not overlay:
        raise SystemExit("nothing to compile")

    entries = merge(esco, overlay)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "_comment": "Generated by tools/build_taxonomy.py - do not hand-edit. "
                    "Edit overlay_skills.json and rebuild.",
        "esco_entries": len(esco), "overlay_entries": len(overlay), "skills": entries,
        "certification_skill_map": load_overlay_cert_map(args.overlay),
    }, ensure_ascii=False), encoding="utf-8")

    print(f"esco:    {len(esco):>6}\noverlay: {len(overlay):>6}", file=sys.stderr)
    print(f"total:   {len(entries):>6} concepts", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)

    if args.collisions:
        clashes = report_collisions(entries)
        print(f"\n{len(clashes)} ambiguous surface form(s) within ESCO.", file=sys.stderr)
        print("The runtime refuses to link these rather than guessing; add an "
              "overlay entry to disambiguate any that matter.", file=sys.stderr)
        for key, ids in sorted(clashes.items())[:40]:
            print(f"  {key!r} -> {len(ids)} concepts", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
