# Resume Screening Agent — Step 1 (L0–L2)

Digital PDF or CSV in → validated `CandidateProfile` JSON out, ready for the
Skill Matching Agent. **Scoped to IT/technology roles.**

```bash
pip install -r requirements.txt
python -m pytest tests -q               # 59 passed
```

## Pipeline

```
0 ingest            PDF/CSV -> text + tokens + geometry
1 segment           header dictionary, or LLM for non-standard headings
2 locate entities   education (3 layouts), work history
3 extract mentions  gazetteer scan (see note below - this is not a learned NER model)
4 semantic filter   reject negated/aspirational/JD terms; redirect objects of analysis
5 link              exact -> alias -> acronym -> fuzzy -> embedding -> unresolved
6 aggregate         merge by skill_id, evidence counts
7 verify            re-read every bbox from the PDF
8 validate          logical checks, parsing_status
```

## Scope: IT roles

The architecture is domain-agnostic; the **knowledge layer** is what narrows.
The overlay covers languages, frameworks, databases, cloud, DevOps, data
engineering, ML/AI, security, mobile, testing and tooling — the vocabulary of
Software Engineer, Data Scientist, ML Engineer, DevOps, Cloud and Cybersecurity
roles. ESCO remains the canonical base if you ever widen scope again.

## Narratives: `projects` and role `description` (schema 1.1)

`skills` is normalised, deduplicated and taxonomy-linked - exactly what the
Matching Agent needs, and exactly what makes it useless for conversation.
"Python, Docker, PyTorch" gives an interviewer nothing specific to ask about.

So the profile also carries what the candidate *built*, in their own words:

```json
"projects": [
  {"title": "Banking Web Application",
   "description": "Developed a secure banking web application using PHP, MySQL and JS with no pre-built frameworks for CS6903: Network Security. Implemented secure session management.",
   "start_date": null, "end_date": null}
],
"work_history": [
  {"company": "Microsoft", "title": "Cloud Engineer",
   "start_date": "2022-01", "end_date": "Present",
   "description": "Managed Azure infrastructure and built CI/CD pipelines for 12 services."}
]
```

Rules:

- **Verbatim, never paraphrased.** No model rewrites these. A generated summary
  cannot be cited back to the candidate, and a recruiter could not verify it.
- **Truncated visibly.** Long descriptions cut at a word boundary and end with
  `…`, so nothing looks complete when it is not.
- **Each role gets only its own bullets**, matched by company or title. An
  unmatched role keeps `description: null` rather than borrowing a neighbour's.
- **A lone "Ongoing"/"Present" is an end date**, not a start - reading it as a
  start would claim the work began today.
- No projects section means `projects: []` plus `PROJECTS_EMPTY`. That is
  absence, not a parse defect, so it does not flip `parsing_status`.

## Terminology (used precisely)

**Stage 3 is skill/entity mention extraction, not NER.** It is a longest-match
n-gram gazetteer scan: it finds terms already in the taxonomy. That trades
recall for precision, which is the right trade here — and it is why *taxonomy
coverage*, not model quality, is the limiting factor on how many skills are
found. A trained NER model would be a different component; we do not claim one.

**The LLM is not a deterministic component.** The defensible claim is narrower:
its output is constrained to line ranges, schema-validated, cached by content
hash, and never used for scoring or ranking. The `CandidateProfile` is
reproducible because every value in it is computed by deterministic code
downstream — not because the model is.

**`candidate_id` is byte-level identity, not duplicate detection.** SHA-256 over
the file bytes makes reprocessing of identical input idempotent. The same resume
re-exported, or with different PDF metadata, hashes differently. Content-level
dedup needs a separate fingerprint over normalised text.

## Stage 4 — semantic filter

The gazetteer answers "does this term appear?", which is not "does this
candidate have this skill".

```
"No production experience with Kubernetes"  -> rejected  (negation)
"Currently learning Rust"                   -> rejected  (aspiration)
"Required skills: Python, Flask"            -> rejected  (requirement)
"Analysed Java malware samples"             -> REDIRECTED: drop Java,
                                               record Malware Analysis
"Built production services in Python"       -> kept
```

**Object-of-analysis redirects rather than discards.** An earlier version
rejected the whole mention, which threw away the capability the sentence
demonstrates. Now the object is dropped and the activity skill is recorded in
its place via a verb-object map, logged as `MENTION_REDIRECTED`. The evidence
bbox still points at the originating sentence, so it verifies and stays
explainable.

Every rejection and redirect is reported. Disable with `--no-semantic-filter` to
measure the layer's effect.

## Stage 5 — linking, with ambiguity refusal

"First match wins" is wrong when a surface form is claimed by two concepts. `ML`
could be Machine Learning or Markup Language; silently taking whichever was
indexed first yields a confidently wrong `skill_id`.

A match is accepted only if it is **unique within its rung**. Ambiguous surface
forms fall through to unresolved and are logged. Fuzzy matching additionally
requires a margin over the runner-up; embedding matching requires both a
threshold and a margin.

The one deliberate exception is **source precedence**: an overlay entry beats an
ESCO entry for the same surface form. That is curation, not ambiguity.

```bash
python -m resume_parser.pipeline --taxonomy-info   # includes ambiguous_surfaces
```

## Taxonomy: ESCO base + IT overlay

```bash
python tools/build_taxonomy.py --esco-skills skills_en.csv \
    --hierarchy skillsHierarchy_en.csv \
    --out resume_parser/data/taxonomy_compiled.json --collisions
```

| | ESCO | Overlay |
|---|---|---|
| Role | canonical cross-sector vocabulary | extension layer |
| `skill_id` | `http://data.europa.eu/esco/skill/<uuid>` | `local:pytorch` |
| Covers | occupational competences | tools, brands, acronyms, emerging tech |
| Maintained by | European Commission | you |

The 123-entry overlay is a **development and extension layer, not the complete
skill vocabulary**. Real coverage comes from ESCO plus overlay. Grow the overlay
from `--unresolved` output, not from memory:

```bash
python -m resume_parser.pipeline resumes/ --unresolved
```

## The labelling loop — do this before adding features

```bash
mkdir gold && cp your_resumes/*.pdf gold/
python tools/make_gold.py gold/          # parser proposes a draft per resume
# edit each gold/<name>.expected.json, set "reviewed": true
python tools/make_gold.py gold/ --status
python tools/evaluate.py --gold gold/    # precision, recall, F1, per file
```

Drafts arrive pre-filled with parser output plus raw text, so labelling is
correcting rather than typing. **Read the resume, not the draft** — approving
drafts unread measures the parser against itself. Unreviewed files are skipped;
reviewed files are never overwritten.

Target: ~30 resumes spread across Software Engineer, Data Scientist, ML
Engineer, Backend, Frontend, DevOps, Cloud and Security roles. Two resumes do
not establish robustness. Fix `POLICY.md` first.

## Design rules enforced in code

| Rule | Where |
|---|---|
| LLM restricted to section line ranges; never scores or ranks | `pipeline.py`, `llm/segmenter.py` |
| Ambiguous surface forms refused, never guessed | `normalization/taxonomy.py` |
| Object of analysis redirects to the activity, not discarded | `extraction/semantics.py` |
| Rejections and redirects reported, never silent | `extraction/semantics.py` |
| `evidence_count` = distinct source types | `schemas` |
| Employment dates beat any prose claim | `pipeline.py` |
| Project duration is not professional experience | `normalization/dates.py` |
| An institution belongs to exactly one degree | `extraction/entities.py` |
| Every bbox re-read from the PDF | `evidence/aggregator.py` |
| Missing field → `null` / `[]` + warning, never fabricated | `validation/validators.py` |

## Known limitations

- **Coordinate clustering is not implemented.** Entity association still counts
  lines, which is why layout quirks keep appearing. This is the next structural
  fix, not another patch — but measure first, so you can prove it helped.
- Institutions and employers are not normalised, so `IITH` and `IIT Hyderabad`
  remain distinct to the Matching Agent.
- Rotated text is neither detected nor handled.
- CSV rows have no page geometry, so `bbox` is `null`.
- Scanned PDFs are out of scope.

## Handoff

```
Resume Screening Agent -> CandidateProfile -> Matching Agent
```

Once precision/recall on the gold set are stable, freeze this and move to the
Matching Agent. Further preprocessing work has diminishing returns.
