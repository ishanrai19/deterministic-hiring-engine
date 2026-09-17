"""Skill linking ladder: exact -> alias -> acronym -> fuzzy -> embedding -> unresolved.

Ambiguity handling
------------------
"First match wins" is wrong when a surface form is claimed by more than one
concept. "ML" can mean Machine Learning or Markup Language; blindly taking
whichever was indexed first produces a confidently wrong skill_id, and nothing
downstream can tell it was wrong.

So the rule is: at each rung, a match is accepted only if it is *unique* within
that rung. If two or more concepts claim the same surface form, the term is
ambiguous and falls through to UNRESOLVED, where it is logged for review.

The one deliberate exception is source precedence: an overlay entry beats an
ESCO entry for the same surface form. That is a curation decision, not
ambiguity - the overlay exists precisely to override.

Rungs 1-4 are offline and always available. Rung 5 (embedding) needs
`sentence-transformers`; without it the taxonomy degrades to rungs 1-4 rather
than failing.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

from rapidfuzz import fuzz, process

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COMPILED_PATH = DATA_DIR / "taxonomy_compiled.json"
OVERLAY_PATH = DATA_DIR / "overlay_skills.json"

FUZZY_THRESHOLD = 92
FUZZY_MARGIN = 4          # winner must beat the runner-up by this many points
FUZZY_MIN_LENGTH = 5
MAX_NGRAM = 6

EMBED_THRESHOLD = 0.62
EMBED_MARGIN = 0.05
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_CASE_SENSITIVE = {"c", "r"}
_CASE_SENSITIVE_ALLOWED = {"C", "R"}


@dataclass(frozen=True)
class SkillMatch:
    skill_id: str
    skill_name: str
    match_rung: str          # exact | alias | acronym | fuzzy | embedding


def _norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^\w#+./\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" .-")


class SkillTaxonomy:
    def __init__(self, path: Optional[Path] = None, *, embeddings: bool = False,
                 embed_model: str = DEFAULT_EMBED_MODEL):
        self.path = Path(path) if path else self._default_path()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.entries: List[dict] = raw["skills"]
        self.certification_skill_map = {_norm(k): v for k, v in
                                        raw.get("certification_skill_map", {}).items()}
        self.esco_entries = raw.get("esco_entries", 0)
        self.overlay_entries = raw.get("overlay_entries", len(self.entries))

        self.by_id: Dict[str, dict] = {e["skill_id"]: e for e in self.entries}

        # Owners per surface form per rung, so ambiguity is visible instead of
        # being silently collapsed by dict assignment.
        owners: Dict[str, Dict[str, Set[str]]] = {
            "exact": defaultdict(set), "alias": defaultdict(set), "acronym": defaultdict(set)}
        overlay_claims: Dict[str, Set[str]] = defaultdict(set)

        for entry in self.entries:
            sid = entry["skill_id"]
            is_overlay = entry.get("source") == "overlay"
            surfaces = []
            if entry.get("name_searchable", True):
                surfaces.append(("exact", entry["skill_name"]))
            surfaces += [("alias", a) for a in entry.get("aliases", [])]
            surfaces += [("acronym", a) for a in entry.get("acronyms", [])]
            for rung, label in surfaces:
                key = _norm(label)
                if not key:
                    continue
                owners[rung][key].add(sid)
                if is_overlay:
                    overlay_claims[key].add(sid)

        # Overlay precedence: where the overlay claims a surface form, it is the
        # only owner. Ambiguity is then judged among the remaining candidates.
        self._owners: Dict[str, Dict[str, Set[str]]] = {}
        for rung, table in owners.items():
            resolved: Dict[str, Set[str]] = {}
            for key, ids in table.items():
                claimed = overlay_claims.get(key)
                resolved[key] = set(claimed) if claimed else set(ids)
            self._owners[rung] = resolved

        self.ambiguous_surfaces: Set[str] = {
            key for table in self._owners.values()
            for key, ids in table.items() if len(ids) > 1}

        # Flat gazetteer for the scanner: any surface worth looking at, including
        # ambiguous ones (they are rejected later, at link time, with a reason).
        self._surface: Dict[str, Set[str]] = defaultdict(set)
        for table in self._owners.values():
            for key, ids in table.items():
                self._surface[key] |= ids
        self._surface.pop("", None)
        self._surface_keys = list(self._surface)
        self.max_ngram = min(max((len(k.split()) for k in self._surface_keys), default=1),
                             MAX_NGRAM)

        self.embed_model_name = embed_model
        self._embedder = None
        self._embed_matrix = None
        self._embed_ids: List[str] = []
        self.embeddings_enabled = self._build_embeddings() if embeddings else False

    @staticmethod
    def _default_path() -> Path:
        if COMPILED_PATH.exists():
            return COMPILED_PATH
        if OVERLAY_PATH.exists():
            return OVERLAY_PATH
        raise FileNotFoundError("no taxonomy found; run tools/build_taxonomy.py")

    # ---------------------------------------------------------- embeddings

    def _build_embeddings(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return False
        import numpy as np

        self._embedder = SentenceTransformer(self.embed_model_name)
        labels = [e["skill_name"] for e in self.entries]
        self._embed_ids = [e["skill_id"] for e in self.entries]
        self._embed_matrix = np.asarray(
            self._embedder.encode(labels, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32")
        return True

    def _embedding_match(self, phrase: str) -> Optional[SkillMatch]:
        if not self.embeddings_enabled or self._embed_matrix is None:
            return None
        import numpy as np

        vector = self._embedder.encode([phrase], normalize_embeddings=True,
                                       show_progress_bar=False)
        scores = self._embed_matrix @ np.asarray(vector, dtype="float32")[0]
        order = np.argsort(-scores)
        best = float(scores[order[0]])
        runner_up = float(scores[order[1]]) if len(order) > 1 else 0.0
        if best < EMBED_THRESHOLD or (best - runner_up) < EMBED_MARGIN:
            return None
        return self._match(self._embed_ids[int(order[0])], "embedding")

    # ---------------------------------------------------------- linking

    def _unique_owner(self, rung: str, key: str) -> Optional[str]:
        ids = self._owners.get(rung, {}).get(key)
        if not ids or len(ids) > 1:
            return None          # unclaimed, or ambiguous -> fall through
        return next(iter(ids))

    def link(self, phrase: str, *, original: Optional[str] = None) -> Optional[SkillMatch]:
        key = _norm(phrase)
        if not key:
            return None
        if key in _CASE_SENSITIVE:
            probe = (original or phrase).strip(" ,;:|/()[]{}.")
            if probe not in _CASE_SENSITIVE_ALLOWED:
                return None

        for rung in ("exact", "alias", "acronym"):
            sid = self._unique_owner(rung, key)
            if sid:
                return self._match(sid, rung)

        if key in self.ambiguous_surfaces:
            return None          # claimed by several concepts; refuse to guess

        if len(key) >= FUZZY_MIN_LENGTH:
            candidates = process.extract(key, self._surface_keys,
                                         scorer=fuzz.token_sort_ratio, limit=2)
            if candidates and candidates[0][1] >= FUZZY_THRESHOLD:
                runner_up = candidates[1][1] if len(candidates) > 1 else 0
                # Two taxonomy entries scoring alike means the term is genuinely
                # ambiguous; only a clear winner is accepted.
                if candidates[0][1] - runner_up >= FUZZY_MARGIN or len(candidates) == 1:
                    owners = self._surface.get(candidates[0][0], set())
                    if len(owners) == 1:
                        return self._match(next(iter(owners)), "fuzzy")
            if self.embeddings_enabled:
                return self._embedding_match(phrase)
        return None

    def _match(self, skill_id: str, rung: str) -> SkillMatch:
        return SkillMatch(skill_id, self.by_id[skill_id]["skill_name"], rung)

    # ---------------------------------------------------------- helpers

    def surface_form_lengths(self) -> range:
        return range(self.max_ngram, 0, -1)

    def lookup_surface(self, phrase: str) -> Optional[str]:
        """Gazetteer hit for the scanner. Returns an id only when unambiguous."""
        ids = self._surface.get(_norm(phrase))
        if not ids or len(ids) > 1:
            return None
        return next(iter(ids))

    def skills_for_certification(self, line: str) -> List[str]:
        key = _norm(line)
        return [sid for cert, ids in self.certification_skill_map.items()
                if cert in key for sid in ids]

    def stats(self) -> dict:
        return {"path": str(self.path), "concepts": len(self.entries),
                "esco_entries": self.esco_entries, "overlay_entries": self.overlay_entries,
                "surface_forms": len(self._surface),
                "ambiguous_surfaces": len(self.ambiguous_surfaces),
                "max_ngram": self.max_ngram,
                "embeddings_enabled": self.embeddings_enabled}


@lru_cache(maxsize=4)
def get_taxonomy(embeddings: bool = False) -> SkillTaxonomy:
    return SkillTaxonomy(embeddings=embeddings)
