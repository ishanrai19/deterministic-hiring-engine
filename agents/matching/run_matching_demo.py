"""
run_matching_demo.py
---------------------
Run from the project root:

    python -m agents.matching.run_matching_demo

What it does:
  1. Loads data/raw/ats_resume_dataset_sample.tsv into CandidateProfile /
     JobDescription lists (label columns kept separate, per the data
     contract's leakage rule).
  2. Runs every candidate against the job it applied to using the default
     "weighted_heuristic" scoring_method and prints a RankedCandidate report.
  3. Trains "supervised_ranker" on this data (shortlisted as target, the
     four engineered features as input) and reports train/test accuracy,
     AUC, and correlation of fit_score with final_score -- exactly the
     evaluation the contract doc specifies ("use shortlisted and
     final_score, not skill_match_score").
  4. Re-scores every candidate with all three scoring_methods side by side.
  5. Ranks candidates within each job_role bucket (contract's documented
     workaround for degenerate per-job ranking).
  6. Writes data/processed/matching_results.csv for the dashboard/eval report.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from agents.matching.matching_agent import MatchingAgent
from utils.data_loader import load_candidate_from_screening_json, load_dataset_csv
from utils.supervised_ranker import SupervisedRanker

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# RAW_PATH = os.path.join(ROOT, "data", "raw", "ats_resume_dataset_sample.tsv")
RAW_PATH = os.path.join(ROOT, "data", "raw", "ats_resume_dataset_elite_v3.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")


def section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def run_weighted_heuristic(agent, candidates, jobs):
    section("1) RANKED CANDIDATE REPORTS -- scoring_method='weighted_heuristic'")
    jobs_by_id = {j["job_id"]: j for j in jobs}
    for c in candidates:
        job = jobs_by_id[c["applied_job_id"]]
        r = agent.match(c, job)
        print(f"\n[{r['candidate_id']}] -> {r['job_title']}  fit_score={r['fit_score']}  "
              f"shortlisted={r['shortlisted']}  (job_id={r['job_id']})")
        print(f"   skill_overlap.overlap_pct={r['skill_overlap']['overlap_pct']}  "
              f"missing={r['skill_overlap']['missing_skills']}")
        for line in r["supporting_evidence"]:
            print(f"   - {line}")


def train_supervised_ranker(candidates, jobs, labels_by_candidate_id):
    section("2) TRAINING scoring_method='supervised_ranker'  (eval vs. shortlisted & final_score)")
    labels = [labels_by_candidate_id.get(c["candidate_id"], 0) for c in candidates]
    ranker = SupervisedRanker()
    metrics = ranker.fit(candidates, {j["job_id"]: j for j in jobs}, labels)
    print(f"Trained on {metrics['n_train']} rows, evaluated on {metrics['n_test']} held-out rows.")
    print(f"  test_accuracy: {metrics.get('test_accuracy')}")
    print(f"  test_auc: {metrics.get('test_auc', 'n/a (single class in test split)')}")
    print(f"  fit_score vs. final_score correlation: {metrics.get('fit_score_vs_final_score_corr', 'n/a')}")
    print("  (NOTE: this sample dataset has only 11 rows -- these numbers are illustrative, "
          "not statistically meaningful. Re-run against the full 6,000-row "
          "ats_resume_dataset_elite_v3.csv for real evaluation numbers.)")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    model_path = os.path.join(PROCESSED_DIR, "supervised_ranker.json")
    ranker.save(model_path)
    print(f"Saved trained ranker -> {model_path}")
    return ranker


def compare_scoring_methods(candidates, jobs, ranker):
    section("3) SIDE-BY-SIDE COMPARISON OF ALL THREE scoring_methods")
    jobs_by_id = {j["job_id"]: j for j in jobs}
    agent = MatchingAgent(supervised_ranker=ranker)

    rows = []
    for c in candidates:
        job = jobs_by_id[c["applied_job_id"]]
        row = {"candidate_id": c["candidate_id"], "job_role": job["title"]}
        for method in ("weighted_heuristic", "embedding_similarity", "supervised_ranker"):
            r = agent.match(c, job, scoring_method=method)
            row[f"fit_score_{method}"] = r["fit_score"]
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df, agent


def rank_within_roles(agent, candidates, jobs):
    section("4) CANDIDATE RANKING WITHIN EACH job_role BUCKET (weighted_heuristic)")
    ranked_by_role = agent.rank_by_role(candidates, jobs)
    for role, ranked in ranked_by_role.items():
        print(f"\n-- {role} --")
        for r in ranked:
            print(f"  #{r['rank']}  {r['candidate_id']:<6} fit_score={r['fit_score']:<7} shortlisted={r['shortlisted']}")


def run_on_screening_agent_example(agent):
    section("5) MATCHING AGAINST A RESUME-SCREENING-AGENT-STYLE JSON PROFILE")
    screening_output = {
        "candidate_id": "CAND_0001",
        "source_resume_path": "resumes/john_doe_resume.pdf",
        "education": [{"degree": "B.Tech", "field": "Computer Science", "institution": "ABC University", "year": 2021}],
        "experience_years": 4.2,
        "work_history": [
            {"company": "Microsoft", "title": "Cloud Engineer", "start_date": "2022-01", "end_date": "Present"},
        ],
        "skills": [
            {"skill_id": "ESCO_AZURE_001", "skill_name": "Azure", "raw_text": "Azure",
             "sources": ["skills_section", "project", "certification", "experience"], "evidence_count": 4},
            {"skill_id": "ESCO_PYTHON_001", "skill_name": "Python", "raw_text": "Python",
             "sources": ["skills_section", "project"], "evidence_count": 2},
            {"skill_id": "ESCO_SQL_001", "skill_name": "SQL", "raw_text": "SQL",
             "sources": ["skills_section"], "evidence_count": 1},
        ],
        "parsing_status": "success",
        "schema_version": "1.0",
        "parsing_timestamp": "2026-09-15T11:00:00Z",
    }
    job = {
        "job_id": "job_cloud_eng_01",
        "title": "Cloud Engineer",
        "required_skills": ["Azure", "Python", "SQL", "Kubernetes"],
        "preferred_skills": [],
        "min_experience_years": 3,
        "min_education": "Bachelors",
        "raw_text": "We are hiring a Cloud Engineer. Required experience: 3 years. Skills required: Azure, Python, SQL, Kubernetes.",
    }
    candidate = load_candidate_from_screening_json(screening_output, applied_job_id=job["job_id"])
    result = agent.match(candidate, job)
    import json
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    candidates, jobs, labels_by_candidate_id = load_dataset_csv(RAW_PATH)

    base_agent = MatchingAgent()
    run_weighted_heuristic(base_agent, candidates, jobs)

    ranker = train_supervised_ranker(candidates, jobs, labels_by_candidate_id)
    comparison_df, full_agent = compare_scoring_methods(candidates, jobs, ranker)

    rank_within_roles(base_agent, candidates, jobs)
    run_on_screening_agent_example(full_agent)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "matching_results.csv")
    comparison_df.to_csv(out_path, index=False)
    print(f"\nSaved comparison table -> {out_path}")
