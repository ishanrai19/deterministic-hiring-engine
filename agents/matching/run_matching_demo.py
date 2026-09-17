"""
run_matching_demo.py
---------------------
Run from the project root:

    # small bundled sample (11 rows)
    python -m agents.matching.run_matching_demo

    # the real dataset
    python -m agents.matching.run_matching_demo --data-path data/raw/ats_resume_dataset_elite_v3.csv

What it does:
  1. Loads the dataset into CandidateProfile / JobDescription lists
     (label columns kept separate, per the data contract's leakage rule).
  2. Prints a handful of individual RankedCandidate reports
     (--print-limit, default 5) using scoring_method='weighted_heuristic'.
  3. Trains scoring_method='supervised_ranker' on the full dataset
     (shortlisted as target, the four engineered features as input) and
     reports train/test accuracy, AUC, and correlation of fit_score with
     final_score -- exactly the evaluation the contract doc specifies.
  4. Scores every row with all three scoring_methods and writes the full
     table to data/processed/matching_results.csv.
  5. Prints per-job_role shortlist-rate summaries (not a full per-candidate
     dump -- there could be thousands of rows).

CLI flags:
  --data-path      Path to the dataset CSV/TSV (default: bundled sample).
  --print-limit N  How many individual candidate reports to print (default 5).
  --shortlist-threshold X   Override the default 0.5 cutoff.
  --output-dir     Where to write matching_results.csv / supervised_ranker.json
                    (default: data/processed).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd

from agents.matching.matching_agent import MatchingAgent
from utils.data_loader import load_candidate_from_screening_json, load_dataset_csv
from utils.supervised_ranker import SupervisedRanker

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SAMPLE_PATH = os.path.join(ROOT, "data", "raw", "ats_resume_dataset_sample.tsv")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "data", "processed")


def section(title: str):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_sample_reports(agent, candidates, jobs, limit):
    section(f"1) SAMPLE RANKED CANDIDATE REPORTS (first {limit}) -- scoring_method='weighted_heuristic'")
    jobs_by_id = {j["job_id"]: j for j in jobs}
    for c in candidates[:limit]:
        job = jobs_by_id[c["applied_job_id"]]
        r = agent.match(c, job)
        print(f"\n[{r['candidate_id']}] -> {r['job_title']}  fit_score={r['fit_score']}  "
              f"shortlisted={r['shortlisted']}  (job_id={r['job_id']})")
        print(f"   skill_overlap.overlap_pct={r['skill_overlap']['overlap_pct']}  "
              f"missing={r['skill_overlap']['missing_skills']}")
        for line in r["supporting_evidence"]:
            print(f"   - {line}")


def train_supervised_ranker(candidates, jobs, labels_by_candidate_id, output_dir):
    section("2) TRAINING scoring_method='supervised_ranker'  (eval vs. shortlisted & final_score)")
    labels = [labels_by_candidate_id.get(c["candidate_id"], 0) for c in candidates]
    ranker = SupervisedRanker()
    metrics = ranker.fit(candidates, {j["job_id"]: j for j in jobs}, labels)
    print(f"Trained on {metrics['n_train']} rows, evaluated on {metrics['n_test']} held-out rows.")
    print(f"  test_accuracy: {metrics.get('test_accuracy')}")
    print(f"  test_auc: {metrics.get('test_auc', 'n/a (single class in test split)')}")
    print(f"  fit_score vs. final_score correlation: {metrics.get('fit_score_vs_final_score_corr', 'n/a')}")
    if metrics["n_train"] + metrics["n_test"] < 100:
        print("  (NOTE: small dataset -- these numbers are illustrative, not statistically meaningful.)")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "supervised_ranker.json")
    ranker.save(model_path)
    print(f"Saved trained ranker -> {model_path}")
    return ranker


def score_all_and_save(candidates, jobs, ranker, output_dir):
    section(f"3) SCORING ALL {len(candidates)} CANDIDATES WITH ALL THREE scoring_methods")
    jobs_by_id = {j["job_id"]: j for j in jobs}
    agent = MatchingAgent(supervised_ranker=ranker)

    rows = []
    for c in candidates:
        job = jobs_by_id[c["applied_job_id"]]
        row = {"candidate_id": c["candidate_id"], "job_role": job["title"]}
        for method in ("weighted_heuristic", "embedding_similarity", "supervised_ranker"):
            r = agent.match(c, job, scoring_method=method)
            row[f"fit_score_{method}"] = r["fit_score"]
            row[f"shortlisted_{method}"] = r["shortlisted"]
        row["skill_overlap_pct"] = agent.match(c, job)["skill_overlap"]["overlap_pct"]
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "matching_results.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")
    return df, agent


def print_role_summary(df):
    section("4) SHORTLIST-RATE SUMMARY BY job_role (weighted_heuristic)")
    summary = (
        df.groupby("job_role")
        .agg(
            n=("candidate_id", "count"),
            avg_fit_score=("fit_score_weighted_heuristic", "mean"),
            shortlist_rate=("shortlisted_weighted_heuristic", "mean"),
        )
        .round(4)
        .sort_values("n", ascending=False)
    )
    print(summary.to_string())


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
    print(json.dumps(result, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Run the Matching Agent end-to-end on a dataset.")
    p.add_argument("--data-path", default=DEFAULT_SAMPLE_PATH,
                    help="Path to the dataset CSV/TSV (default: bundled 11-row sample).")
    p.add_argument("--print-limit", type=int, default=5,
                    help="How many individual candidate reports to print (default: 5).")
    p.add_argument("--shortlist-threshold", type=float, default=0.5,
                    help="fit_score cutoff for shortlisted=True (default: 0.5).")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help="Where to write matching_results.csv / supervised_ranker.json.")
    p.add_argument("--skip-screening-example", action="store_true",
                    help="Skip the demo match against a Resume-Screening-Agent-style JSON profile.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.data_path):
        print(f"ERROR: dataset not found at '{args.data_path}'.")
        print("Pass the real dataset with, e.g.:")
        print("  python -m agents.matching.run_matching_demo "
              "--data-path data/raw/ats_resume_dataset_elite_v3.csv")
        sys.exit(1)

    print(f"Loading dataset: {args.data_path}")
    candidates, jobs, labels_by_candidate_id = load_dataset_csv(args.data_path)
    print(f"Loaded {len(candidates)} candidates / {len(jobs)} job postings "
          f"({len(labels_by_candidate_id)} with a shortlisted label).")

    base_agent = MatchingAgent(shortlist_threshold=args.shortlist_threshold)
    print_sample_reports(base_agent, candidates, jobs, args.print_limit)

    ranker = train_supervised_ranker(candidates, jobs, labels_by_candidate_id, args.output_dir)
    results_df, full_agent = score_all_and_save(candidates, jobs, ranker, args.output_dir)

    print_role_summary(results_df)

    if not args.skip_screening_example:
        run_on_screening_agent_example(full_agent)

    print(f"\nDone. Full results: {os.path.join(args.output_dir, 'matching_results.csv')}")
