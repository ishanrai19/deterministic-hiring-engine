"""
matching_agent.py
------------------
Matching Agent (P2) -- job-requirement comparison, match score, candidate
ranking, skill-gap analysis, supporting evidence.

Output shape follows the team's RankedCandidate contract exactly:

    {
      "candidate_id": ..., "job_id": ...,
      "skill_overlap": {"matched_skills": [...], "missing_skills": [...], "overlap_pct": 0.5},
      "fit_score": 0.0-1.0,
      "rank": int | None,
      "scoring_method": "weighted_heuristic" | "embedding_similarity" | "supervised_ranker",
      ... plus extra explainability fields (experience_detail, education_detail,
          supporting_evidence) for the dashboard / Interview Agent, which the
          contract doesn't forbid but doesn't require either.
    }

Three interchangeable scoring_methods, all built on the same four
features (utils.scoring.build_feature_vector), so they can be compared
apples-to-apples:

  - "weighted_heuristic" (default): transparent weighted sum. No training
    data required, works cold-start on day one, every coefficient is a
    number a recruiter can be shown and challenged on.
  - "embedding_similarity": leans on resume<->JD semantic similarity plus
    skill overlap, no experience/education weighting. Useful when the JD
    text is rich and skill lists are noisy/incomplete.
  - "supervised_ranker": a logistic regression trained on the dataset's
    `shortlisted` label (utils.supervised_ranker.SupervisedRanker), used
    the way the contract specifies -- labels for training/eval ONLY, the
    trained model takes only the four features at inference time.

DATA LEAKAGE: this module never reads shortlisted / final_score /
skill_match_score / experience_match / education_match / similarity_score
from a job or candidate dict when computing a match. Those only ever
appear (if at all) under job['_reference_scores'], which
utils.scoring.build_feature_vector() ignores by construction.
"""

from __future__ import annotations

import datetime as _dt
from typing import Dict, List, Optional

from utils.scoring import build_feature_vector
from utils.scoring import semantic_similarity
from utils.supervised_ranker import SupervisedRanker

DEFAULT_WEIGHTS = {
    "skills": 0.40,
    "experience": 0.25,
    "education": 0.15,
    "semantic": 0.20,
}

SCHEMA_VERSION = "1.0"
VALID_SCORING_METHODS = {"weighted_heuristic", "embedding_similarity", "supervised_ranker"}


class MatchingAgent:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        shortlist_threshold: float = 0.5,
        default_scoring_method: str = "weighted_heuristic",
        supervised_ranker: Optional[SupervisedRanker] = None,
    ):
        weights = weights or DEFAULT_WEIGHTS
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Match-score weights must sum to 1.0 (got {total}).")
        if default_scoring_method not in VALID_SCORING_METHODS:
            raise ValueError(f"scoring_method must be one of {VALID_SCORING_METHODS}")

        self.weights = weights
        self.shortlist_threshold = shortlist_threshold
        self.default_scoring_method = default_scoring_method
        self.supervised_ranker = supervised_ranker

    # ------------------------------------------------------------------ #
    def match(self, candidate: Dict, job: Dict, scoring_method: Optional[str] = None) -> Dict:
        """Compare one candidate against one JobDescription -> one RankedCandidate."""
        scoring_method = scoring_method or self.default_scoring_method
        if scoring_method not in VALID_SCORING_METHODS:
            raise ValueError(f"scoring_method must be one of {VALID_SCORING_METHODS}")

        vector, details = build_feature_vector(candidate, job)
        skill_overlap_pct, experience_ratio, education_ratio, semantic_score = vector
        skills_detail = details["skills"]
        experience_detail = details["experience"]
        education_detail = details["education"]

        fit_score = self._compute_fit_score(scoring_method, vector)
        shortlisted = fit_score >= self.shortlist_threshold

        return {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": candidate.get("candidate_id"),
            "job_id": job.get("job_id"),
            "job_title": job.get("title"),
            "skill_overlap": {
                "matched_skills": skills_detail["matched_skills"],
                "matched_skills_detail": skills_detail["matched_skills_detail"],
                "missing_skills": skills_detail["missing_skills"],
                "extra_skills": skills_detail["extra_skills"],
                "overlap_pct": skill_overlap_pct,
            },
            "fit_score": fit_score,
            "rank": None,  # populated by rank_candidates()
            "scoring_method": scoring_method,
            "shortlisted": shortlisted,
            "experience_detail": experience_detail,
            "education_detail": education_detail,
            "supporting_evidence": self._build_evidence(
                skills_detail, experience_detail, education_detail, semantic_score
            ),
            "matched_at": _dt.datetime.utcnow().isoformat() + "Z",
        }

    def _compute_fit_score(self, scoring_method: str, vector: List[float]) -> float:
        skill_overlap_pct, experience_ratio, education_ratio, semantic_score = vector

        if scoring_method == "weighted_heuristic":
            return round(
                self.weights["skills"] * skill_overlap_pct
                + self.weights["experience"] * experience_ratio
                + self.weights["education"] * education_ratio
                + self.weights["semantic"] * semantic_score,
                4,
            )

        if scoring_method == "embedding_similarity":
            # Leans on semantic + raw skill overlap; ignores experience/education
            # weighting, useful when the JD text carries most of the signal.
            return round(0.6 * semantic_score + 0.4 * skill_overlap_pct, 4)

        if scoring_method == "supervised_ranker":
            if self.supervised_ranker is None:
                raise RuntimeError(
                    "scoring_method='supervised_ranker' requires a trained "
                    "SupervisedRanker to be passed into MatchingAgent(supervised_ranker=...)."
                )
            return self.supervised_ranker.predict_fit_score(vector)

        raise ValueError(f"Unknown scoring_method: {scoring_method}")

    # ------------------------------------------------------------------ #
    # Candidate ranking for a single job posting (the well-defined case)
    # ------------------------------------------------------------------ #
    def rank_candidates(self, candidates: List[Dict], job: Dict, scoring_method: Optional[str] = None) -> List[Dict]:
        """Match every candidate against `job` and return results sorted
        best-first, each annotated with its rank."""
        results = [self.match(c, job, scoring_method=scoring_method) for c in candidates]
        results.sort(key=lambda r: r["fit_score"], reverse=True)
        for i, r in enumerate(results, start=1):
            r["rank"] = i
        return results

    # ------------------------------------------------------------------ #
    # Batch: each candidate matched only against the job it applied for
    # ------------------------------------------------------------------ #
    def batch_match(self, candidates: List[Dict], jobs: List[Dict], scoring_method: Optional[str] = None) -> List[Dict]:
        jobs_by_id = {j["job_id"]: j for j in jobs}
        results = []
        for c in candidates:
            job = jobs_by_id.get(c.get("applied_job_id"))
            if job is None:
                continue
            results.append(self.match(c, job, scoring_method=scoring_method))
        return results

    def rank_by_role(
        self, candidates: List[Dict], jobs: List[Dict], scoring_method: Optional[str] = None
    ) -> Dict[str, List[Dict]]:
        """
        Per the team's contract note: 'job_description is unique per row, so
        within-job ranking is degenerate. Group by job_role (6 buckets) and
        treat the task as pointwise shortlist prediction.' This ranks each
        role's candidates by fit_score computed against their OWN applied
        job (pointwise), not against one shared posting -- the ranking
        within a role bucket is still meaningful even though no two
        candidates in this dataset compete for literally the same posting.
        """
        jobs_by_id = {j["job_id"]: j for j in jobs}
        results = self.batch_match(candidates, jobs, scoring_method=scoring_method)

        by_role: Dict[str, List[Dict]] = {}
        for r in results:
            by_role.setdefault(r["job_title"], []).append(r)

        for role, role_results in by_role.items():
            role_results.sort(key=lambda r: r["fit_score"], reverse=True)
            for i, r in enumerate(role_results, start=1):
                r["rank"] = i

        return by_role

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_evidence(skills_detail, experience_detail, education_detail, semantic_score) -> List[str]:
        evidence = []

        if skills_detail["matched_skills"]:
            evidence.append(
                f"Matched {skills_detail['matched_count']}/{skills_detail['required_count']} "
                f"required skills: {', '.join(skills_detail['matched_skills'])}."
            )
        if skills_detail["missing_skills"]:
            evidence.append(f"Missing required skills: {', '.join(skills_detail['missing_skills'])}.")

        if experience_detail["meets_requirement"]:
            evidence.append(
                f"Meets experience requirement "
                f"({experience_detail['candidate_years']} yrs >= {experience_detail['required_years']} yrs)."
            )
        else:
            evidence.append(
                f"Below experience requirement by {experience_detail['gap_years']} yrs "
                f"({experience_detail['candidate_years']} yrs / {experience_detail['required_years']} yrs required)."
            )

        if education_detail["required_level"]:
            status = "meets" if education_detail["meets_requirement"] else "below"
            evidence.append(
                f"Education '{education_detail['candidate_level']}' {status} "
                f"requirement '{education_detail['required_level']}'."
            )

        evidence.append(f"Resume-to-JD semantic similarity: {semantic_score}.")
        return evidence
