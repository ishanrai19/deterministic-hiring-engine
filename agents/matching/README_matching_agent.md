# Matching Agent (P2)

Owner: Ishan
Deliverables: job-requirement comparison, match score, candidate ranking, skill-gap analysis, supporting evidence.

## Does this need "agentic AI"?

No. This repo is `deterministic-hiring-engine`, and the team's data
contract specifies `scoring_method: embedding_similarity | supervised_ranker`
for the Matching Agent — both are quantitative/statistical techniques,
not an LLM reasoning loop. LLM usage is scoped to the **Interview Agent**
(explicitly the "LLM-powered Question Generator") and the **LLM
Selection** workstream. Keeping Matching deterministic and label-free at
inference time is a feature, not a gap: it's fast, reproducible, and a
recruiter can be shown exactly why a candidate scored the way they did.

## Output: RankedCandidate

`MatchingAgent().match(candidate, job)` returns:

```json
{
  "candidate_id": "R0", "job_id": "job_dbd44ec93c", "job_title": "Software Engineer",
  "skill_overlap": {
    "matched_skills": ["Machine Learning", "Power BI", "Python"],
    "missing_skills": ["C++", "Data Analysis", "Flask", "TensorFlow"],
    "overlap_pct": 0.4286
  },
  "fit_score": 0.6231,
  "rank": null,
  "scoring_method": "weighted_heuristic",
  "shortlisted": true,
  "experience_detail": {...}, "education_detail": {...},
  "supporting_evidence": ["Matched 3/7 required skills: ...", "..."]
}
```

This matches the team's contract field names exactly
(`skill_overlap.{matched_skills,missing_skills,overlap_pct}`, `fit_score`,
`rank`, `scoring_method`), with `experience_detail` / `education_detail` /
`supporting_evidence` added as extra explainability fields for the
dashboard and Interview Agent — the contract doesn't forbid extra fields.

`job_id` is derived from a hash of `job_description` (per the contract's
"derived (hash of job_description) — not a column"), not a synthetic
`JOB_{resume_id}` placeholder like the previous version used.

## Three scoring_methods, one feature set

All three read the same four features
(`utils.scoring.build_feature_vector`): skill overlap %, experience ratio,
education-level ratio, semantic similarity. This keeps them directly
comparable and keeps the label boundary in one place.

| scoring_method | How fit_score is computed | When to use |
|---|---|---|
| `weighted_heuristic` (default) | `0.40·skills + 0.25·experience + 0.15·education + 0.20·semantic` | Cold start, no labels needed, fully explainable coefficients |
| `embedding_similarity` | `0.6·semantic + 0.4·skill_overlap` | JD text is rich, skill lists noisy/incomplete |
| `supervised_ranker` | Logistic regression trained on `shortlisted`, `predict_proba` | When enough labeled data exists to train/validate against |

**Data-leakage boundary (enforced in code, not just by convention):**
`build_feature_vector()` never reads `skill_match_score`, `experience_match`,
`education_match`, `final_score`, `shortlisted`, or `similarity_score` —
those live only under `job['_reference_scores']`, which the scoring
functions structurally ignore. `SupervisedRanker.fit()` is the *only*
place `shortlisted` is read, and only as a training target; its
`predict_fit_score()` at inference time takes just the four features.
`test_matching_agent_never_reads_label_columns` asserts this holds.

`SupervisedRanker` evaluation reports test accuracy, AUC, and the
correlation of `fit_score` against `final_score` — exactly what the
contract specifies ("use shortlisted and final_score, not
skill_match_score, that is just exact overlap and any model would tie
it"). On the 11-row sample dataset these numbers are illustrative only;
re-run against the full 6,000-row `ats_resume_dataset_elite_v3.csv` for
numbers worth reporting in the Evaluation Report.

## Two bugs found and fixed in this pass

1. **Silent taxonomy fallback.** After the project was restructured into
   `agents/matching/`, the skills-taxonomy JSON file wasn't found at
   runtime, and `SkillNormalizer` silently fell back to naive title-casing
   instead of warning — producing "Power Bi", "Node.Js", "Sql" instead of
   "Power BI", "Node.js", "SQL". Fixed: `SkillNormalizer` now (a) searches
   several plausible paths + an env var override, (b) hardcodes the
   dataset's closed 19-skill vocabulary as a safety net so normalization
   is correct even if the JSON file is never found, and (c) raises a
   `UserWarning` when it can't find the file, so this fails loudly instead
   of quietly degrading.
2. **Acronym-casing bug in the fallback itself.** The old fallback checked
   `key.isupper()` where `key` was already lowercased by `_clean()` — that
   condition could never be true, so *every* unmatched skill got
   title-cased regardless of whether it was an acronym. Fixed: the check
   now runs against the original, un-lowercased string.

Both are covered by new tests (`test_normalizer_preserves_acronyms`,
`test_normalizer_survives_missing_taxonomy_file`).

## Files

| File | Purpose |
|---|---|
| `agents/matching/matching_agent.py` | `MatchingAgent`: match, rank_candidates, batch_match, rank_by_role. |
| `agents/matching/run_matching_demo.py` | `python -m agents.matching.run_matching_demo` — end-to-end demo + ranker training. |
| `utils/scoring.py` | Skill/experience/education scoring + `build_feature_vector()`. |
| `utils/skill_normalizer.py` | Alias/acronym-safe skill canonicalization (bugfixed). |
| `utils/supervised_ranker.py` | Logistic-regression `scoring_method='supervised_ranker'`. |
| `utils/data_loader.py` | JobDescription/CandidateProfile adapters, `derive_job_id()`. |
| `data/skills_taxonomy.json` | Extendable taxonomy (not required for this closed-vocab dataset, but ready for real resumes). |
| `data/raw/ats_resume_dataset_sample.tsv` | The provided sample rows, for local testing. |
| `data/processed/` | Demo run outputs: `matching_results.csv`, `supervised_ranker.json`. |
| `tests/test_matching_agent.py` | 14 unit tests, including the two bugfix regressions and the leakage-boundary check. |

## Running it

```bash
cd deterministic-hiring-engine
pip install pandas scikit-learn numpy
python -m agents.matching.run_matching_demo
python tests/test_matching_agent.py
```

## Known limitations / open items for the team

- `min_education` isn't a real dataset column (contract doc: "implied by
  education_match threshold"). `weighted_heuristic`/`embedding_similarity`
  treat "no requirement" as auto-pass for that feature; `supervised_ranker`
  can learn whatever implicit threshold the labels encode, since education
  rank is one of its four input features.
- Per the contract's "Limitation" note, job postings in this dataset are
  unique per row even when `job_role` repeats, so `rank_by_role()` ranks
  each role's candidates pointwise against their own applied job, not
  against one shared posting — consistent with the contract's suggested
  workaround (treat as per-role shortlist prediction, not literal
  within-job ranking).
- The 19-skill closed vocabulary is hardcoded in `skill_normalizer.py` as
  a safety net. If the Dataset workstream's real taxonomy diverges from
  the sample dataset (e.g. for the full 6,000-row dataset or real-world
  resumes), point `MATCHING_AGENT_TAXONOMY_PATH` at the shared file or
  extend `data/skills_taxonomy.json`.
