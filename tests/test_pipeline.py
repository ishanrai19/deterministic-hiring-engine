"""Run with: python -m pytest tests -q"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import fitz
import pytest

from resume_parser.extraction.sections import classify_line
from resume_parser.extraction.semantics import classify_mention
from resume_parser.ingestion.loaders import load_pdf
from resume_parser.llm.segmenter import SegmentCache, parse_response, validate_sections
from resume_parser.normalization.dates import (
    experience_years, find_date_range, find_stated_experience, normalise_date)
from resume_parser.normalization.taxonomy import SkillTaxonomy, get_taxonomy
from resume_parser.pipeline import LLMConfig, parse_csv, parse_document, parse_pdf

TODAY = date(2026, 9, 17)
ROOT = Path(__file__).resolve().parent.parent
OVERLAY = ROOT / "resume_parser" / "data" / "overlay_skills.json"


def _pdf(tmp_path, text, name="r.pdf"):
    path = tmp_path / name
    doc = fitz.open()
    page = doc.new_page()
    y = 55
    for line in text.split("\n"):
        page.insert_text((60, y), line, fontsize=10)
        y += 15
    doc.save(path)
    doc.close()
    return str(path)


RESUME = """John Doe

SKILLS
Python, SQL, Azure, Docker

EXPERIENCE
Cloud Engineer | Microsoft | Jan 2022 - Present
Systems Engineer | Infosys Limited | Jun 2020 - Dec 2021

PROJECTS
Customer Churn Prediction - built with Azure ML and Python

CERTIFICATIONS
Microsoft Azure AI Engineer Associate

EDUCATION
B.Tech in Computer Science
ABC University, 2020
"""

BACKWARD_EDU = """EDUCATION
Indian Institute of Technology, Hyderabad
2024 - Present
M.Tech in Computer Science and Engineering
Chitkara University, Punjab
2019 - 2023
B.Tech in Computer Science and Engineering

SKILLS
Others: C++, DSA
"""


@pytest.fixture(scope="module")
def resume_pdf(tmp_path_factory):
    return _pdf(tmp_path_factory.mktemp("pdf"), RESUME)


# ------------------------------------------------------------------ dates


def test_normalise_date_variants():
    assert normalise_date("January 2022") == "2022-01"
    assert normalise_date("Jan '22") == "2022-01"
    assert normalise_date("July-2025") == "2025-07"
    assert normalise_date("2021") == "2021"
    assert normalise_date("Till Date") == "Present"
    assert normalise_date("gibberish") is None


def test_date_range():
    assert find_date_range("Cloud Engineer | Jan 2022 - Present") == ("2022-01", "Present")


def test_overlapping_experience_is_not_double_counted():
    assert experience_years([("2023-01", "2023-12"), ("2023-07", "2023-12")], today=TODAY)[0] == 1.0


def test_stated_experience_requires_the_word_experience():
    assert find_stated_experience("Candidate with 10 years of experience.") == 10.0
    assert find_stated_experience("8+ years of experience") == 8.0
    assert find_stated_experience("200 years of experience") is None


def test_project_duration_is_not_professional_experience():
    """'Built a system over 3 years' is a project duration, not a career total."""
    assert find_stated_experience("Built a distributed system over 3 years") is None
    assert find_stated_experience("Worked on the platform for 2 years") is None


# ------------------------------------------------------------------ semantics


@pytest.mark.parametrize("sentence,term,accepted", [
    ("No production experience with Kubernetes yet.", "Kubernetes", False),
    ("Currently learning Rust for systems work.", "Rust", False),
    ("Seeking to learn Go in my next role.", "Go", False),
    ("Required skills: Python, Flask.", "Python", False),
    ("Built production services in Python.", "Python", True),
    ("Deployed with Docker and Kubernetes.", "Kubernetes", True),
])
def test_semantic_cues(sentence, term, accepted):
    start = sentence.index(term)
    assert classify_mention(sentence, start, start + len(term), "experience").accepted is accepted


def test_object_of_analysis_redirects_rather_than_discarding():
    """The object is not the skill, but the activity around it still is.

    Rejecting the whole mention threw away the capability the sentence
    demonstrates. It must redirect instead.
    """
    text = "Analysed Java malware samples in the lab."
    start = text.index("Java")
    verdict = classify_mention(text, start, start + 4, "experience")
    assert verdict.accepted is False
    assert verdict.reason == "object"
    assert verdict.redirect_surface == "malware analysis"


def test_redirect_produces_the_activity_skill_end_to_end(tmp_path):
    path = _pdf(tmp_path, "SKILLS\nPython, Docker\n\nEXPERIENCE\n"
                          "Security Analyst | Zeta Labs | Jan 2022 - Present\n"
                          "Analysed Java malware samples in the lab.\n")
    on = {s.skill_name for s in parse_pdf(path, today=TODAY, semantic_filter=True).profile.skills}
    off = {s.skill_name for s in parse_pdf(path, today=TODAY, semantic_filter=False).profile.skills}

    assert "Java" in off                      # naive scan credits the object
    assert "Java" not in on                   # it is subject matter, not a skill
    assert "Malware Analysis" in on           # but the activity survives
    assert "Malware Analysis" not in off


def test_redirect_is_reported_not_silent(tmp_path):
    path = _pdf(tmp_path, "EXPERIENCE\nAnalysed Java malware samples.\n\nSKILLS\nPython\n")
    result = parse_pdf(path, today=TODAY)
    assert any(w.startswith("MENTION_REDIRECTED") for w in result.warnings)


def test_negation_does_not_leak_across_clauses():
    text = "Strong in Python. Not familiar with Rust."
    start = text.index("Python")
    assert classify_mention(text, start, start + 6, "skills_section").accepted


def test_filter_outcomes_do_not_flip_parsing_status():
    from resume_parser.validation.validators import INFO_CODES
    assert {"MENTION_REJECTED", "MENTION_REDIRECTED"} <= INFO_CODES


# ------------------------------------------------------------------ taxonomy ambiguity


def _tax_with(entries, tmp_path):
    path = tmp_path / "tax.json"
    path.write_text(json.dumps({"skills": entries, "certification_skill_map": {}}))
    return SkillTaxonomy(path)


def test_ambiguous_surface_form_is_refused_not_guessed(tmp_path):
    """'ML' claimed by two concepts must not silently resolve to whichever was
    indexed first."""
    tax = _tax_with([
        {"skill_id": "x:ml", "skill_name": "Machine Learning", "acronyms": ["ml"],
         "source": "esco"},
        {"skill_id": "x:markup", "skill_name": "Markup Language", "acronyms": ["ml"],
         "source": "esco"},
    ], tmp_path)
    assert "ml" in tax.ambiguous_surfaces
    assert tax.link("ML") is None
    assert tax.lookup_surface("ML") is None


def test_unambiguous_surface_still_links(tmp_path):
    tax = _tax_with([
        {"skill_id": "x:ml", "skill_name": "Machine Learning", "acronyms": ["ml"],
         "source": "esco"},
    ], tmp_path)
    assert tax.link("ML").skill_id == "x:ml"


def test_overlay_precedence_is_not_treated_as_ambiguity(tmp_path):
    """Overlay overriding ESCO is curation, not ambiguity - it must still link."""
    tax = _tax_with([
        {"skill_id": "esco:py", "skill_name": "Python", "aliases": ["python"],
         "source": "esco"},
        {"skill_id": "local:python", "skill_name": "Python", "aliases": ["python"],
         "source": "overlay"},
    ], tmp_path)
    assert tax.link("Python").skill_id == "local:python"
    assert "python" not in tax.ambiguous_surfaces


def test_linking_ladder():
    tax = get_taxonomy()
    assert tax.link("Python").match_rung == "exact"
    assert tax.link("MS Azure").skill_id == "local:azure"
    assert tax.link("quantum basket weaving") is None


def test_single_letter_skills_require_upper_case():
    tax = get_taxonomy()
    assert tax.link("C", original="C").skill_id == "local:c"
    assert tax.link("c", original="c") is None


def test_androguard_is_malware_analysis_not_android_development():
    assert get_taxonomy().link("Androguard").skill_id == "local:malware_analysis"


def test_embeddings_are_optional_and_off_by_default():
    tax = SkillTaxonomy(OVERLAY)
    assert tax.stats()["embeddings_enabled"] is False
    assert tax.link("Python") is not None


def test_embedding_request_degrades_when_library_absent():
    tax = SkillTaxonomy(OVERLAY, embeddings=True)
    assert isinstance(tax.embeddings_enabled, bool)   # degraded, never raised
    assert tax.link("Python") is not None


def test_it_overlay_covers_target_roles():
    """The overlay is the IT extension layer; ESCO remains the canonical base."""
    tax = get_taxonomy()
    for term in ["React", "Kubernetes", "PyTorch", "Terraform", "Kafka", "GraphQL",
                 "Penetration Testing", "MLOps", "System Design", "Spring Boot"]:
        assert tax.link(term) is not None, term


# ------------------------------------------------------------------ sections


def test_colon_headers_are_recognised():
    assert classify_line("Skills:")[0] == "skills_section"
    assert classify_line("Professional Summary:")[0] == "other"


def test_comma_list_is_not_a_header():
    assert classify_line("NLP, SQL, C++") == (None, False)


def test_all_caps_unknown_header_resets_section():
    assert classify_line("PROJECTS") == ("project", True)
    assert classify_line("RELEVANT COURSES") == ("other", True)
    assert classify_line("SOME UNKNOWN BANNER") == (None, True)


def test_all_caps_line_with_a_year_is_content():
    assert classify_line("ICAR IARI 2020") == (None, False)


# ------------------------------------------------------------------ education


def test_institution_above_degree_is_attributed_correctly(tmp_path):
    profile = parse_pdf(_pdf(tmp_path, BACKWARD_EDU), today=TODAY).profile
    pairs = [(e.degree, e.institution) for e in profile.education]
    assert ("M.Tech", "Indian Institute of Technology, Hyderabad") in pairs
    assert ("B.Tech", "Chitkara University, Punjab") in pairs


def test_institution_below_degree_still_works(tmp_path):
    path = _pdf(tmp_path, "EDUCATION\nM.Tech Computer Science\nIIT Hyderabad\n2026\n"
                          "B.Tech Electronics\nIIIT Sricity\n2023\n\nSKILLS\nPython\n")
    pairs = [(e.degree, e.institution, e.year)
             for e in parse_pdf(path, today=TODAY).profile.education]
    assert ("M.Tech", "IIT Hyderabad", 2026) in pairs
    assert ("B.Tech", "IIIT Sricity", 2023) in pairs


def test_prose_education_line(tmp_path):
    path = _pdf(tmp_path, "EDUCATION\nB.Tech in Computer Science\nABC University, 2020\n\nSKILLS\nPython\n")
    edu = parse_pdf(path, today=TODAY).profile.education[0]
    assert (edu.degree, edu.institution, edu.year) == ("B.Tech", "ABC University", 2020)


def test_school_rows_are_skipped(tmp_path):
    path = _pdf(tmp_path, "EDUCATION\nB.Tech Computer Science\nABC University, 2021\n"
                          "XII (Telangana State Board)\nDeeksha College\n2019\n\nSKILLS\nPython\n")
    assert [e.degree for e in parse_pdf(path, today=TODAY).profile.education] == ["B.Tech"]


# ------------------------------------------------------------------ skills lines


def test_label_prefix_is_not_scanned_as_a_skill(tmp_path):
    path = _pdf(tmp_path, "SKILLS\nMachine Learning: Numpy, Pandas, Sklearn\n"
                          "Programming Language: C, C++, Python\n")
    names = {s.skill_name for s in parse_pdf(path, today=TODAY).profile.skills}
    assert {"NumPy", "Pandas", "scikit-learn", "C", "C++", "Python"} <= names
    assert "Machine Learning" not in names


def test_hyphen_wrapped_skill_is_rejoined(tmp_path):
    path = _pdf(tmp_path, "SKILLS\nSkills Learnt: Chat Application using Socket Pro-\ngramming\n")
    assert "Socket Programming" in {s.skill_name for s in parse_pdf(path, today=TODAY).profile.skills}


def test_courses_section_is_not_skill_evidence(tmp_path):
    path = _pdf(tmp_path, "PROJECTS\nBuilt a parser in Python\n\nRELEVANT COURSES\nDeep Learning\n\nSKILLS\nPython\n")
    names = {s.skill_name for s in parse_pdf(path, today=TODAY).profile.skills}
    assert "Deep Learning" not in names and "Python" in names


# ------------------------------------------------------------------ gold tooling


def test_make_gold_creates_editable_draft(tmp_path):
    _pdf(tmp_path, RESUME, name="cand.pdf")
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_gold.py"), str(tmp_path)],
                   check=True, capture_output=True, cwd=ROOT)
    payload = json.loads((tmp_path / "cand.expected.json").read_text())
    assert payload["reviewed"] is False
    assert "skills" in payload and "_raw_text" in payload


def test_make_gold_never_overwrites_reviewed_work(tmp_path):
    _pdf(tmp_path, RESUME, name="cand.pdf")
    gold = tmp_path / "cand.expected.json"
    gold.write_text(json.dumps({"reviewed": True, "skills": ["Python"], "education": [],
                                "experience_years": None, "work_history": []}))
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_gold.py"),
                    str(tmp_path), "--refresh"], check=True, capture_output=True, cwd=ROOT)
    assert json.loads(gold.read_text())["skills"] == ["Python"]


def test_evaluate_ignores_unreviewed_drafts(tmp_path):
    _pdf(tmp_path, RESUME, name="cand.pdf")
    (tmp_path / "cand.expected.json").write_text(json.dumps(
        {"reviewed": False, "skills": ["Python"], "education": [],
         "experience_years": None, "work_history": []}))
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "evaluate.py"),
                          "--gold", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert "no reviewed gold files" in out.stdout


def test_evaluate_scores_reviewed_gold(tmp_path):
    _pdf(tmp_path, RESUME, name="cand.pdf")
    (tmp_path / "cand.expected.json").write_text(json.dumps({
        "reviewed": True,
        "skills": ["Python", "SQL", "Microsoft Azure", "Docker"],
        "education": [{"degree": "B.Tech", "institution": "ABC University"}],
        "experience_years": 6.3,
        "work_history": [{"company": "Microsoft"}]}))
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "evaluate.py"),
                          "--gold", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert "precision" in out.stdout and "reviewed resumes: 1" in out.stdout


# ------------------------------------------------------------------ LLM


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return json.dumps(self.payload) if isinstance(self.payload, dict) else self.payload


def _llm_config(client, cache=None):
    cfg = LLMConfig(enabled=True)
    cfg._client = client
    cfg._cache = cache
    return cfg


ODD = """Arjun Mehta

What I Know
Python, Docker, Kubernetes, PostgreSQL

Jobs Done
Backend Developer | Zeta Systems | Mar 2021 - Present

Papers I Studied
Deep Learning
"""

ODD_SECTIONS = {"sections": [
    {"section": "other", "start_line": 0, "end_line": 0},
    {"section": "skills_section", "start_line": 1, "end_line": 2},
    {"section": "experience", "start_line": 3, "end_line": 4},
    {"section": "other", "start_line": 5, "end_line": 7},
]}


def test_llm_handles_nonstandard_headings(tmp_path):
    doc = load_pdf(_pdf(tmp_path, ODD, name="odd.pdf"))
    result = parse_document(doc, today=TODAY, llm=_llm_config(FakeLLM(ODD_SECTIONS)))
    assert result.segmenter == "llm"
    assert "PostgreSQL" in {s.skill_name for s in result.profile.skills}


@pytest.mark.parametrize("payload", [
    "not json at all",
    {"sections": [{"section": "salary", "start_line": 0, "end_line": 3}]},
    {"sections": [{"section": "experience", "start_line": 5, "end_line": 2}]},
    {"sections": [{"section": "experience", "start_line": 0, "end_line": 5},
                  {"section": "project", "start_line": 3, "end_line": 8}]},
    {"sections": []},
])
def test_bad_llm_output_falls_back(tmp_path, payload):
    doc = load_pdf(_pdf(tmp_path, ODD, name="odd.pdf"))
    result = parse_document(doc, today=TODAY, llm=_llm_config(FakeLLM(payload)))
    assert result.segmenter == "deterministic"
    assert any(w.startswith("LLM_FALLBACK") for w in result.warnings)
    assert result.profile.skills


def test_llm_transport_failure_falls_back(tmp_path):
    class Boom:
        def complete(self, system, user):
            raise RuntimeError("connection refused")

    doc = load_pdf(_pdf(tmp_path, ODD, name="odd.pdf"))
    result = parse_document(doc, today=TODAY, llm=_llm_config(Boom()))
    assert result.segmenter == "deterministic"
    assert any(w.startswith("LLM_UNAVAILABLE") for w in result.warnings)


def test_llm_response_is_cached_by_content_hash(tmp_path):
    client = FakeLLM(ODD_SECTIONS)
    cache = SegmentCache(tmp_path / "cache")
    doc = load_pdf(_pdf(tmp_path, ODD, name="odd.pdf"))
    first = parse_document(doc, today=TODAY, llm=_llm_config(client, cache))
    second = parse_document(doc, today=TODAY, llm=_llm_config(client, cache))
    assert client.calls == 1
    assert first.profile.parsing_status == second.profile.parsing_status


def test_llm_markdown_fences_are_tolerated():
    raw = '```json\n{"sections": [{"section": "other", "start_line": 0, "end_line": 1}]}\n```'
    assert parse_response(raw)[0]["section"] == "other"


def test_validate_clamps_end_line_within_document():
    assert validate_sections([{"section": "experience", "start_line": 0, "end_line": 999}],
                             10) == [("experience", 0, 9)]


def test_llm_disabled_by_default(resume_pdf):
    assert parse_pdf(resume_pdf, today=TODAY).segmenter == "deterministic"


# ------------------------------------------------------------------ pipeline


def test_end_to_end(resume_pdf):
    profile = parse_pdf(resume_pdf, today=TODAY).profile
    by_name = {s.skill_name: s for s in profile.skills}
    azure = by_name["Microsoft Azure"]
    assert set(azure.sources) >= {"skills_section", "certification"}
    assert azure.evidence_count == len(set(azure.sources))
    assert profile.education[0].degree == "B.Tech"
    assert profile.experience_years == pytest.approx(6.3, abs=0.2)


def test_dated_history_wins_over_stated_total(tmp_path):
    path = _pdf(tmp_path, "SKILLS\nPython\n\nEXPERIENCE\nCandidate with 30 years of experience.\n"
                          "Data Engineer | Acme Ltd | Jan 2024 - Dec 2024\n")
    assert parse_pdf(path, today=TODAY).profile.experience_years == pytest.approx(1.0, abs=0.1)


def test_missing_education_is_empty_list_not_invented(tmp_path):
    path = _pdf(tmp_path, "SKILLS\nPython\n\nEXPERIENCE\nData Engineer | Acme Ltd | Jan 2021 - Present\n")
    result = parse_pdf(path, today=TODAY)
    assert result.profile.education == []
    assert any(w.startswith("EDUCATION_EMPTY") for w in result.warnings)


def test_empty_pdf_fails_cleanly(tmp_path):
    path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    assert parse_pdf(str(path), today=TODAY).profile.parsing_status == "failed"


def test_evidence_count_equals_distinct_sources(resume_pdf):
    for skill in parse_pdf(resume_pdf, today=TODAY).profile.skills:
        assert skill.evidence_count == len(set(skill.sources))
        assert {l.source for l in skill.locations} <= set(skill.sources)


def test_bboxes_reread_to_the_same_text(resume_pdf):
    profile = parse_pdf(resume_pdf, today=TODAY).profile
    with fitz.open(resume_pdf) as pdf:
        for skill in profile.skills:
            for loc in skill.locations:
                rect = fitz.Rect(loc.bbox.x0 - 1, loc.bbox.y0 - 1, loc.bbox.x1 + 1, loc.bbox.y1 + 1)
                assert (loc.raw_text or skill.skill_name).lower()[:8] in \
                    pdf[loc.page - 1].get_textbox(rect).lower()


def test_csv_rows_have_no_bounding_boxes(tmp_path):
    import pandas as pd
    path = tmp_path / "r.csv"
    pd.DataFrame({"resume_id": ["R0"],
                  "resume_text": ["Skills:\nPython, SQL\n"]}).to_csv(path, index=False)
    profile = list(parse_csv(path, text_column="resume_text", today=TODAY))[0].profile
    for skill in profile.skills:
        for loc in skill.locations:
            assert loc.bbox is None


def test_identical_bytes_give_identical_ids(resume_pdf, tmp_path):
    """Byte-level idempotency. NOT content-level duplicate detection."""
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(open(resume_pdf, "rb").read())
    assert load_pdf(resume_pdf).candidate_id == load_pdf(copy).candidate_id


# ------------------------------------------------------------------ narratives


BULLET_PROJECTS = """EDUCATION
B.Tech in Computer Science
ABC University, 2020

PROJECTS
Banking Web Application (TransactiWar)
- Developed a secure banking web application using PHP, MySQL and JS with no
pre-built frameworks for CS6903: Network Security.
- Implemented secure session management.
Daily Journal
- Technologies used: NodeJs, Express, EJS.
- A personalized diary where users store everyday activities.

EXPERIENCE
Cloud Engineer | Microsoft | Jan 2022 - Present
- Managed Azure infrastructure and built CI/CD pipelines for 12 services.
Systems Engineer | Infosys Limited | Jun 2020 - Dec 2021
- Automated deployments using Docker and Linux shell scripting.
"""

PIPE_PROJECTS = """PROJECTS
Uncertainty Estimation in LLM | LLM Reliability | (Guide: Prof. Srijith)
M.Tech Thesis - Ongoing
Developing a novel method to estimate uncertainty in large language models using
a single forward pass, reducing cost compared to Bayesian dropout.
Motiverse AI 2025 IITH Hackathon | NLP, RAG, LangChain (Github)
August - 2025
Built a multi-turn RAG chatbot using AWS Bedrock and LangChain.

SKILLS
Python, PyTorch
"""


def test_projects_capture_title_and_description(tmp_path):
    projects = parse_pdf(_pdf(tmp_path, BULLET_PROJECTS), today=TODAY).profile.projects
    titles = [p.title for p in projects]
    assert "Banking Web Application" in titles       # trailing "(TransactiWar)" stripped
    assert "Daily Journal" in titles
    banking = next(p for p in projects if p.title == "Banking Web Application")
    assert "secure banking web application" in banking.description
    assert "session management" in banking.description


def test_project_description_is_verbatim_not_paraphrased(tmp_path):
    """Descriptions must be quotable back to the candidate."""
    path = _pdf(tmp_path, BULLET_PROJECTS)
    raw = "\n".join(page.get_text() for page in fitz.open(path))
    flat_raw = " ".join(raw.split())
    for project in parse_pdf(path, today=TODAY).profile.projects:
        for fragment in (project.description or "").rstrip("…").split(". ")[:1]:
            assert fragment.strip(" .…") in flat_raw


def test_pipe_delimited_titles_and_metadata_lines(tmp_path):
    projects = parse_pdf(_pdf(tmp_path, PIPE_PROJECTS), today=TODAY).profile.projects
    titles = [p.title for p in projects]
    assert "Uncertainty Estimation in LLM" in titles
    assert "Motiverse AI 2025 IITH Hackathon" in titles
    # "Developing a novel method..." is a description, not a third project.
    assert not any(t.startswith("Developing") for t in titles)


def test_ongoing_alone_is_an_end_date_not_a_start(tmp_path):
    """Reading 'Ongoing' as a start would claim the work began today."""
    project = next(p for p in parse_pdf(_pdf(tmp_path, PIPE_PROJECTS),
                                        today=TODAY).profile.projects
                   if p.title == "Uncertainty Estimation in LLM")
    assert project.start_date is None
    assert project.end_date == "Present"


def test_each_role_gets_its_own_description(tmp_path):
    """A role must never borrow a neighbouring role's bullets."""
    work = parse_pdf(_pdf(tmp_path, BULLET_PROJECTS), today=TODAY).profile.work_history
    by_company = {w.company: w for w in work}
    assert "Azure infrastructure" in by_company["Microsoft"].description
    assert "Docker" in by_company["Infosys Limited"].description
    assert "Azure" not in by_company["Infosys Limited"].description


def test_projects_stay_separate_from_skills(tmp_path):
    """Two different questions: what can they do, versus what did they build."""
    profile = parse_pdf(_pdf(tmp_path, BULLET_PROJECTS), today=TODAY).profile
    assert profile.projects and profile.skills
    assert {s.skill_name for s in profile.skills} & {"PHP", "MySQL", "Docker"}
    assert all(isinstance(p.title, str) for p in profile.projects)


def test_long_description_is_truncated_at_a_word_boundary(tmp_path):
    from resume_parser.extraction.narratives import MAX_DESCRIPTION_CHARS
    # Many short lines, not one long one: insert_text does not wrap, so a
    # single long line would be clipped by the page before truncation applied.
    body = "\n".join("- Built an extremely detailed distributed system component."
                     for _ in range(12))
    path = _pdf(tmp_path, f"PROJECTS\nHuge Project\n{body}\n\nSKILLS\nPython\n")
    description = parse_pdf(path, today=TODAY).profile.projects[0].description
    assert len(description) <= MAX_DESCRIPTION_CHARS + 1
    assert description.endswith("…")        # the cut is visible, not silent
    assert not description.rstrip("…").endswith(" ")


def test_resume_without_projects_reports_empty_not_invented(tmp_path):
    path = _pdf(tmp_path, "SKILLS\nPython\n\nEXPERIENCE\n"
                          "Data Engineer | Acme Ltd | Jan 2021 - Present\n")
    result = parse_pdf(path, today=TODAY)
    assert result.profile.projects == []
    assert any(w.startswith("PROJECTS_EMPTY") for w in result.warnings)


def test_missing_projects_does_not_flip_status(tmp_path):
    """Most resumes have no projects section; that is absence, not a defect."""
    from resume_parser.validation.validators import INFO_CODES
    assert "PROJECTS_EMPTY" in INFO_CODES


def test_schema_version_records_the_new_fields(resume_pdf):
    profile = parse_pdf(resume_pdf, today=TODAY).profile
    assert profile.schema_version == "1.1"
    payload = json.loads(parse_pdf(resume_pdf, today=TODAY).to_json())
    assert "projects" in payload
    assert all("description" in w for w in payload["work_history"])
