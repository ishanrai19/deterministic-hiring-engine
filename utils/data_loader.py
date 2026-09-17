"""
data_loader.py
---------------
Adapters that build the two contract objects the Matching Agent consumes,
per the team's Pipeline Workflow & Data Contracts doc:

  JobDescription:
    { job_id, title, required_skills, preferred_skills,
      min_experience_years, min_education, raw_text }

  CandidateProfile (from the Resume Screening Agent, or adapted from the
  flat sample dataset for local testing):
    { candidate_id, skills, evidence_counts, experience_years,
      education_level, resume_text, applied_job_id }

Two loaders are provided:

  A) load_candidate_from_screening_json() -- adapts the Resume Screening
     Agent's structured JSON output (the real production input).

  B) load_dataset_csv() -- adapts the flat sample/eval dataset
     (ats_resume_dataset_elite_v3.csv-style) into (candidates, jobs).

DATA LEAKAGE BOUNDARY: skill_match_score, experience_match, education_match,
final_score, shortlisted, similarity_score are evaluation labels, per the
contract doc. load_dataset_csv() reads them into a separate
job['_reference_scores'] / a parallel `labels` list -- NEVER into the
CandidateProfile/JobDescription fields that the Matching Agent's scoring
functions read. utils/scoring.build_feature_vector() only reads the
non-label fields, so this boundary is enforced structurally, not just by
convention.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _split_skills(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def derive_job_id(job_description_text: str) -> str:
    """job_id is derived (hash of job_description) -- it is not a dataset column."""
    digest = hashlib.md5((job_description_text or "").encode("utf-8")).hexdigest()
    return f"job_{digest[:10]}"


# --------------------------------------------------------------------------- #
# A) Resume Screening Agent JSON -> CandidateProfile (internal shape)
# --------------------------------------------------------------------------- #
def load_candidate_from_screening_json(profile: Dict, applied_job_id: Optional[str] = None) -> Dict:
    skills = []
    evidence_counts: Dict[str, int] = {}
    for s in profile.get("skills", []):
        name = s.get("skill_name") or s.get("raw_text")
        if not name:
            continue
        skills.append(name)
        evidence_counts[name] = s.get("evidence_count", len(s.get("sources", [])))

    from utils.scoring import education_rank

    education_entries = profile.get("education", []) or []
    best_degree, best_rank = "", -1
    for edu in education_entries:
        degree = edu.get("degree", "")
        rank = education_rank(degree)
        if rank > best_rank:
            best_rank, best_degree = rank, degree

    # Projects carry real semantic signal (what the candidate actually built,
    # in their own words) that skills[] alone doesn't capture -- include them
    # in resume_text so semantic_similarity() can see it. Schema 1.1 adds this
    # field; schema 1.0 profiles simply won't have it (.get returns []).
    project_text_parts = []
    for proj in profile.get("projects", []) or []:
        title = proj.get("title", "")
        description = proj.get("description", "")
        project_text_parts.append(f"{title}. {description}".strip())

    resume_text_parts = [profile.get("source_resume_path", "")] + skills + project_text_parts
    for wh in profile.get("work_history", []) or []:
        resume_text_parts.append(f"{wh.get('title', '')} at {wh.get('company', '')}")

    raw_experience_years = profile.get("experience_years")

    return {
        "candidate_id": profile.get("candidate_id"),
        "applied_job_id": applied_job_id,
        "skills": skills,
        "evidence_counts": evidence_counts,
        # NOTE: if the Screening Agent output experience_years: null (couldn't
        # extract it), this stays None here -- NOT silently defaulted to 0.0.
        # utils.scoring.score_experience() treats None as 0 for the actual
        # score (a conservative default), but MatchingAgent.match() checks
        # this raw value to add an explicit "extraction failed" evidence line
        # so a 0 score isn't mistaken for a verified 0 years of experience.
        "experience_years": raw_experience_years,
        "education_level": best_degree,
        "resume_text": " ".join(p for p in resume_text_parts if p),
        # Pass-through metadata, not used in scoring -- surfaced in match()
        # supporting_evidence and available to the dashboard for a
        # "partial parse, verify manually" flag.
        "parsing_status": profile.get("parsing_status"),
    }


# --------------------------------------------------------------------------- #
# B) Flat CSV dataset -> (CandidateProfile[], JobDescription[])
# --------------------------------------------------------------------------- #
def load_dataset_csv(path: str) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    Load the flat evaluation dataset and split it into:
      - candidates: List[CandidateProfile]  (agent input)
      - jobs:       List[JobDescription]     (agent input; label columns kept
                                               separately under _reference_scores)
      - labels_by_candidate_id: Dict[candidate_id, shortlisted:int]  (eval-only,
                                               kept OUT of the candidate/job dicts)
    """
    df = pd.read_csv(path, sep="\t") if _looks_like_tsv(path) else pd.read_csv(path)

    required_columns = {
        "resume_id", "resume_text", "resume_skills", "experience_years",
        "education_level", "job_role", "required_skills",
        "job_experience_required", "job_description",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"load_dataset_csv: '{path}' is missing expected column(s) {sorted(missing)}. "
            f"Columns found: {list(df.columns)}. "
            "If this is a different dataset version, either rename its columns to match, "
            "or adjust the column names read in this function."
        )

    candidates: List[Dict] = []
    jobs: List[Dict] = []
    labels_by_candidate_id: Dict[str, int] = {}

    for _, row in df.iterrows():
        resume_id = str(row.get("resume_id"))
        job_description_text = str(row.get("job_description", ""))
        job_id = derive_job_id(job_description_text)

        candidates.append(
            {
                "candidate_id": resume_id,
                "applied_job_id": job_id,
                "skills": _split_skills(row.get("resume_skills")),
                "evidence_counts": {},
                "experience_years": float(row.get("experience_years", 0) or 0),
                "education_level": row.get("education_level", ""),
                "resume_text": str(row.get("resume_text", "")),
            }
        )

        jobs.append(
            {
                "job_id": job_id,
                "title": row.get("job_role", ""),
                "required_skills": _split_skills(row.get("required_skills")),
                "preferred_skills": [],  # not present in this dataset
                "min_experience_years": float(row.get("job_experience_required", 0) or 0),
                "min_education": None,  # not present as a column; see contract doc
                "raw_text": job_description_text,
                # EVAL-ONLY reference labels. build_feature_vector() never reads this key.
                "_reference_scores": {
                    "skill_match_score": row.get("skill_match_score"),
                    "experience_match": row.get("experience_match"),
                    "education_match": row.get("education_match"),
                    "final_score": row.get("final_score"),
                    "shortlisted": row.get("shortlisted"),
                    "similarity_score": row.get("similarity_score"),
                },
            }
        )

        shortlisted = row.get("shortlisted")
        if pd.notna(shortlisted):
            labels_by_candidate_id[resume_id] = int(shortlisted)

    return candidates, jobs, labels_by_candidate_id


def _looks_like_tsv(path: str) -> bool:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    return "\t" in first_line and "," not in first_line.split("\t")[0]


def load_json_file(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
