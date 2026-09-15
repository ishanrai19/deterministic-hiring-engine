"""
skill_normalizer.py
--------------------
Normalizes raw skill strings (from resumes or job descriptions) into a
canonical display form so that "ML", "machine learning" and "Machine
Learning" are all recognized as the same skill, and acronyms like "SQL"
or "NLP" don't get mangled into "Sql" / "Nlp".

Per the team's data contract (ats_resume_dataset_elite_v3.csv), the
current dataset uses a CLOSED, already-normalized 19-skill vocabulary
on both the resume and job side, so ESCO/a full taxonomy isn't required
for this dataset. That 19-skill list is hardcoded below as a safety net
(CANONICAL_SKILLS) so normalization is correct even if the taxonomy
JSON file can't be found (e.g. after a project restructure). The JSON
file (data/skills_taxonomy.json) can still be swapped/extended for
real-world resumes later without touching this file.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, Iterable, List, Optional

# Closed vocabulary confirmed against the team's data contract doc
# ("6 technical job roles ... closed 19-skill vocabulary"). Kept as a
# hardcoded safety net independent of whether the taxonomy JSON loads.
CANONICAL_SKILLS: List[str] = [
    "Machine Learning", "Python", "Excel", "Power BI", "PyTorch", "SQL",
    "Docker", "Java", "NLP", "Flask", "Cloud Computing", "C++", "Node.js",
    "Data Analysis", "Deep Learning", "Django", "TensorFlow", "Kubernetes",
    "React",
]

_THIS_FILE = os.path.abspath(__file__)
_UTILS_DIR = os.path.dirname(_THIS_FILE)
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)

# Try several plausible locations so the taxonomy is found regardless of
# how the project gets reorganized (agents/matching/, a different cwd, etc.)
_CANDIDATE_TAXONOMY_PATHS = [
    os.environ.get("MATCHING_AGENT_TAXONOMY_PATH", ""),
    os.path.join(_PROJECT_ROOT, "data", "skills_taxonomy.json"),
    os.path.join(_PROJECT_ROOT, "data", "raw", "skills_taxonomy.json"),
    os.path.join(os.getcwd(), "data", "skills_taxonomy.json"),
]


class SkillNormalizer:
    def __init__(self, taxonomy_path: Optional[str] = None, warn_if_missing: bool = True):
        self.taxonomy_path = taxonomy_path
        self._alias_to_canonical: Dict[str, str] = {}

        # 1. Seed with the hardcoded closed vocabulary (always available,
        #    never depends on file resolution succeeding).
        for name in CANONICAL_SKILLS:
            self._alias_to_canonical[self._clean(name)] = name

        # 2. Try to load/extend from a taxonomy JSON file.
        resolved = self._resolve_taxonomy_path(taxonomy_path)
        if resolved:
            self._load_taxonomy(resolved)
        elif warn_if_missing:
            warnings.warn(
                "SkillNormalizer: no skills_taxonomy.json found in any of "
                f"{[p for p in _CANDIDATE_TAXONOMY_PATHS if p]}. "
                "Falling back to the hardcoded 19-skill closed vocabulary only. "
                "Set MATCHING_AGENT_TAXONOMY_PATH or pass taxonomy_path= explicitly "
                "if you intended to use a custom/extended taxonomy.",
                stacklevel=2,
            )

    @staticmethod
    def _resolve_taxonomy_path(explicit_path: Optional[str]) -> Optional[str]:
        candidates = [explicit_path] if explicit_path else list(_CANDIDATE_TAXONOMY_PATHS)
        for path in candidates:
            if path and os.path.exists(path):
                return path
        return None

    def _load_taxonomy(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            taxonomy = json.load(f)
        for skill_id, entry in taxonomy.items():
            if skill_id.startswith("_"):
                continue
            canonical = entry.get("name", skill_id)
            for alias in entry.get("aliases", []):
                self._alias_to_canonical[self._clean(alias)] = canonical
            self._alias_to_canonical[self._clean(canonical)] = canonical

    @staticmethod
    def _clean(raw: str) -> str:
        return " ".join(raw.strip().lower().split())

    def normalize(self, raw: str) -> str:
        """Return the canonical display name for a raw skill string."""
        if not raw:
            return ""
        key = self._clean(raw)
        if key in self._alias_to_canonical:
            return self._alias_to_canonical[key]

        # Fallback for skills outside the known vocabulary: preserve the
        # original casing if it looks like an acronym (e.g. "AWS", "GCP"),
        # otherwise title-case it. Checked against the ORIGINAL raw text,
        # not the already-lowercased `key` (previous version's bug).
        stripped = raw.strip()
        looks_like_acronym = stripped.isupper() and stripped.isalpha() and len(stripped) <= 5
        return stripped if looks_like_acronym else key.title()

    def normalize_many(self, raw_skills: Iterable[str]) -> List[str]:
        """Normalize a list of raw skill strings, de-duplicated, order-preserving."""
        seen = set()
        out: List[str] = []
        for raw in raw_skills:
            canon = self.normalize(raw)
            if canon and canon not in seen:
                seen.add(canon)
                out.append(canon)
        return out
