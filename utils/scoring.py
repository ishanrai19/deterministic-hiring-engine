"""
scoring.py
----------
Standalone, unit-testable scoring functions used by the Matching Agent:

  1. score_skills()        -> skill-matching logic (matched / missing / extra + score)
  2. score_experience()    -> years-of-experience match
  3. score_education()     -> education-level match
  4. semantic_similarity() -> TF-IDF cosine similarity between resume text and JD text
                               (falls back to Jaccard token overlap if scikit-learn
                               is unavailable, so the agent never hard-fails)

These are kept separate from the MatchingAgent class so they can be
reused, unit tested, and eventually replaced by the shared
"Algorithms" workstream's candidate-scoring formula without touching
the agent orchestration code.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from utils.skill_normalizer import SkillNormalizer

_normalizer = SkillNormalizer()

# Ordinal education scale. Extend as needed; unseen labels default to 0.
EDUCATION_LEVELS: Dict[str, int] = {
    "none": 0,
    "high school": 1,
    "diploma": 2,
    "associate": 2,
    "bachelors": 3,
    "bachelor": 3,
    "b.tech": 3,
    "btech": 3,
    "bsc": 3,
    "b.sc": 3,
    "masters": 4,
    "master": 4,
    "m.tech": 4,
    "mtech": 4,
    "msc": 4,
    "m.sc": 4,
    "mba": 4,
    "phd": 5,
    "ph.d": 5,
    "doctorate": 5,
}


def _clean_level(label: Optional[str]) -> str:
    return (label or "").strip().lower()


def education_rank(label: Optional[str]) -> int:
    """Map a free-text education label to an ordinal rank (0-5)."""
    return EDUCATION_LEVELS.get(_clean_level(label), 0)


# --------------------------------------------------------------------------- #
# 1. Skill matching
# --------------------------------------------------------------------------- #
def score_skills(
    candidate_skills: Iterable[str],
    required_skills: Iterable[str],
    evidence_counts: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    Compare a candidate's skills against a job's required skills.

    Score = (# required skills the candidate has) / (# required skills).
    This rewards candidates for covering the requirement, and does not
    penalize them for having *additional* skills beyond the requirement
    (those are surfaced separately as 'extra_skills' - useful context for
    a recruiter, not a penalty).

    `evidence_counts`, if supplied (skill_name -> evidence_count from the
    Resume Screening Agent output), is attached per matched skill so the
    match result carries traceable evidence rather than a bare boolean.
    """
    candidate_norm = _normalizer.normalize_many(candidate_skills)
    required_norm = _normalizer.normalize_many(required_skills)

    candidate_set = set(candidate_norm)
    required_set = set(required_norm)

    matched = sorted(candidate_set & required_set)
    missing = sorted(required_set - candidate_set)
    extra = sorted(candidate_set - required_set)

    # overlap_pct == skill_match_score: |matched| / |required|. This is the
    # same definition as the dataset's own skill_match_score column, so it
    # should correlate ~1.0 against it once skills are normalized consistently
    # (per the team's data contract note "reproduces skill_match_score, corr 1.0").
    score = round(len(matched) / len(required_set), 4) if required_set else 1.0

    evidence_counts = evidence_counts or {}
    matched_with_evidence = [
        {
            "skill": skill,
            "evidence_count": evidence_counts.get(skill, evidence_counts.get(skill.lower(), None)),
        }
        for skill in matched
    ]

    return {
        "score": score,
        "matched_skills": matched,
        "matched_skills_detail": matched_with_evidence,
        "missing_skills": missing,   # skill gap
        "extra_skills": extra,       # candidate has, JD doesn't ask for
        "required_count": len(required_set),
        "matched_count": len(matched),
    }


# --------------------------------------------------------------------------- #
# 2. Experience matching
# --------------------------------------------------------------------------- #
def score_experience(candidate_years: float, required_years: float) -> Dict:
    """
    1.0  if candidate meets or exceeds the requirement.
    Otherwise partial credit = candidate_years / required_years.
    A job with no experience requirement (<=0) is treated as fully met.
    """
    candidate_years = max(float(candidate_years or 0), 0)
    required_years = float(required_years or 0)

    if required_years <= 0:
        score = 1.0
    elif candidate_years >= required_years:
        score = 1.0
    else:
        score = round(candidate_years / required_years, 4)

    return {
        "score": score,
        "candidate_years": candidate_years,
        "required_years": required_years,
        "meets_requirement": candidate_years >= required_years,
        "gap_years": max(round(required_years - candidate_years, 2), 0),
    }


# --------------------------------------------------------------------------- #
# 3. Education matching
# --------------------------------------------------------------------------- #
def score_education(candidate_level: Optional[str], required_level: Optional[str]) -> Dict:
    """
    1.0 if candidate's education rank >= required rank.
    Partial credit (rank ratio) otherwise.
    No requirement specified -> fully met.
    """
    c_rank = education_rank(candidate_level)
    r_rank = education_rank(required_level)

    if r_rank <= 0:
        score = 1.0
    elif c_rank >= r_rank:
        score = 1.0
    else:
        score = round(c_rank / r_rank, 4) if r_rank else 1.0

    return {
        "score": score,
        "candidate_level": candidate_level,
        "required_level": required_level,
        "meets_requirement": c_rank >= r_rank,
    }


# --------------------------------------------------------------------------- #
# 4. Semantic similarity (resume text vs. job description)
# --------------------------------------------------------------------------- #
def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Cosine similarity between TF-IDF vectors of the two texts.
    Captures contextual/semantic overlap beyond exact skill-string matches
    (e.g. "built scalable APIs" vs. "backend development experience").

    Falls back to a simple Jaccard token-overlap score if scikit-learn is
    not installed, so the agent degrades gracefully rather than crashing.
    """
    text_a = (text_a or "").strip()
    text_b = (text_b or "").strip()
    if not text_a or not text_b:
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return round(float(sim), 4)
    except ImportError:
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return round(len(tokens_a & tokens_b) / len(tokens_a | tokens_b), 4)


# --------------------------------------------------------------------------- #
# 5. Feature vector (shared by the weighted heuristic and the supervised ranker,
#    so both scoring methods reason about the exact same inputs)
# --------------------------------------------------------------------------- #
FEATURE_NAMES = [
    "skill_overlap_pct",
    "experience_ratio",
    "education_rank_ratio",
    "semantic_similarity",
]


def build_feature_vector(candidate: Dict, job: Dict) -> Tuple[List[float], Dict]:
    """
    Compute the four core match features for a candidate/job pair.
    Returns (feature_vector, component_details) where component_details
    carries the full score_skills/score_experience/score_education dicts
    for explainability (skill gaps, evidence, etc).

    IMPORTANT: this function must never read shortlisted / final_score /
    skill_match_score / experience_match / education_match / similarity_score
    from the job or candidate dicts -- those are evaluation labels only,
    per the team's data-leakage rule.
    """
    skills_result = score_skills(
        candidate_skills=candidate.get("skills", []),
        required_skills=job.get("required_skills", []),
        evidence_counts=candidate.get("evidence_counts", {}),
    )
    experience_result = score_experience(
        candidate_years=candidate.get("experience_years", 0),
        required_years=job.get("min_experience_years", job.get("required_experience_years", 0)),
    )
    education_result = score_education(
        candidate_level=candidate.get("education_level", ""),
        required_level=job.get("min_education") or job.get("required_education_level"),
    )
    semantic_score = semantic_similarity(
        text_a=candidate.get("resume_text", ""),
        text_b=job.get("raw_text", job.get("job_description", "")),
    )

    vector = [
        skills_result["score"],
        experience_result["score"],
        education_result["score"],
        semantic_score,
    ]
    details = {
        "skills": skills_result,
        "experience": experience_result,
        "education": education_result,
        "semantic_similarity": semantic_score,
    }
    return vector, details
