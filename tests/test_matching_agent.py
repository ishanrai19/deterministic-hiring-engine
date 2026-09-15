import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.matching.matching_agent import MatchingAgent
from utils.data_loader import derive_job_id
from utils.scoring import score_education, score_experience, score_skills
from utils.skill_normalizer import SkillNormalizer


def test_normalizer_preserves_acronyms():
    n = SkillNormalizer()
    assert n.normalize("SQL") == "SQL"
    assert n.normalize("sql") == "SQL"
    assert n.normalize("Power BI") == "Power BI"
    assert n.normalize("power bi") == "Power BI"
    assert n.normalize("Node.js") == "Node.js"
    assert n.normalize("NLP") == "NLP"


def test_normalizer_survives_missing_taxonomy_file():
    # Must not crash, and must still get the closed-vocabulary skills right
    # even if the taxonomy JSON can't be found on disk (e.g. after a
    # project restructure).
    n = SkillNormalizer(taxonomy_path="/definitely/does/not/exist.json", warn_if_missing=False)
    assert n.normalize("SQL") == "SQL"
    assert n.normalize("TensorFlow".lower()) == "TensorFlow"


def test_score_skills_full_match():
    r = score_skills(["Python", "SQL", "Excel"], ["python", "sql"])
    assert r["score"] == 1.0
    assert r["missing_skills"] == []
    assert "Excel" in r["extra_skills"]


def test_score_skills_partial_match():
    r = score_skills(["Python"], ["Python", "SQL"])
    assert r["score"] == 0.5
    assert r["missing_skills"] == ["SQL"]


def test_score_experience_meets():
    r = score_experience(5, 3)
    assert r["score"] == 1.0
    assert r["meets_requirement"] is True


def test_score_experience_partial():
    r = score_experience(1, 4)
    assert r["score"] == 0.25
    assert r["meets_requirement"] is False


def test_score_education_meets():
    r = score_education("Masters", "Bachelors")
    assert r["score"] == 1.0


def test_score_education_no_requirement():
    r = score_education("Bachelors", None)
    assert r["score"] == 1.0


def test_derive_job_id_is_stable_and_hash_based():
    a = derive_job_id("We are hiring a Data Scientist.")
    b = derive_job_id("We are hiring a Data Scientist.")
    c = derive_job_id("We are hiring a Backend Engineer.")
    assert a == b
    assert a != c
    assert a.startswith("job_")


def _sample_job(job_id="J1"):
    return {
        "job_id": job_id,
        "title": "Data Scientist",
        "required_skills": ["Python", "SQL", "Machine Learning", "TensorFlow"],
        "preferred_skills": [],
        "min_experience_years": 3,
        "min_education": "Bachelors",
        "raw_text": "Looking for a data scientist with Python, SQL, ML and TensorFlow experience.",
    }


def _sample_candidate(candidate_id="C1", applied_job_id="J1"):
    return {
        "candidate_id": candidate_id,
        "applied_job_id": applied_job_id,
        "skills": ["Python", "SQL", "Machine Learning"],
        "evidence_counts": {},
        "experience_years": 5,
        "education_level": "Masters",
        "resume_text": "Experienced data scientist skilled in Python, SQL and machine learning.",
    }


def test_matching_agent_ranked_candidate_schema():
    agent = MatchingAgent()
    result = agent.match(_sample_candidate(), _sample_job())

    # Contract fields must be present with the contract's exact names
    assert result["candidate_id"] == "C1"
    assert result["job_id"] == "J1"
    assert set(["matched_skills", "missing_skills", "overlap_pct"]) <= set(result["skill_overlap"].keys())
    assert result["skill_overlap"]["overlap_pct"] == 0.75
    assert "TensorFlow" in result["skill_overlap"]["missing_skills"]
    assert 0 <= result["fit_score"] <= 1
    assert result["scoring_method"] == "weighted_heuristic"
    assert result["shortlisted"] in (True, False)


def test_matching_agent_never_reads_label_columns():
    agent = MatchingAgent()
    job = _sample_job()
    job["_reference_scores"] = {
        "skill_match_score": 0.0,  # deliberately wrong/misleading
        "final_score": 0.0,
        "shortlisted": 0,
    }
    result = agent.match(_sample_candidate(), job)
    # The presence of (wrong) reference scores on the job dict must not
    # change the computed fit_score -- proves the leakage boundary holds.
    baseline = agent.match(_sample_candidate(), _sample_job())
    assert result["fit_score"] == baseline["fit_score"]


def test_rank_candidates_orders_best_first():
    agent = MatchingAgent()
    job = _sample_job()
    strong = _sample_candidate("strong")
    weak = {
        "candidate_id": "weak",
        "applied_job_id": "J1",
        "skills": ["Excel"],
        "experience_years": 0,
        "education_level": "High School",
        "resume_text": "Office assistant.",
    }
    ranked = agent.rank_candidates([weak, strong], job)
    assert ranked[0]["candidate_id"] == "strong"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["candidate_id"] == "weak"
    assert ranked[1]["rank"] == 2


def test_embedding_similarity_scoring_method_runs():
    agent = MatchingAgent()
    result = agent.match(_sample_candidate(), _sample_job(), scoring_method="embedding_similarity")
    assert result["scoring_method"] == "embedding_similarity"
    assert 0 <= result["fit_score"] <= 1


def test_supervised_ranker_requires_trained_model():
    agent = MatchingAgent()  # no supervised_ranker passed in
    try:
        agent.match(_sample_candidate(), _sample_job(), scoring_method="supervised_ranker")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASSED: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
