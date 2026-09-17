# Labelling policy — IT roles

Decisions the parser cannot make for you. Fix these **before** labelling; your
evaluation numbers are meaningless until the definition is stable.

Scope: IT/technology roles — Software Engineer, Data Scientist, ML Engineer,
Data Analyst, Backend/Frontend Developer, DevOps, Cloud Engineer, Cybersecurity
Analyst, AI/GenAI Engineer.

## What counts as a skill

| Case | Decision | Rationale |
|---|---|---|
| Listed in a Skills section | **yes** | explicit claim |
| Demonstrated in a project | **yes** | evidenced |
| Used in paid work | **yes** | strongest evidence |
| Implied by a certification | **yes** | via `certification_skill_map` |
| Listed under *Courses Taken* | **no** | studying a subject is not a demonstrated skill |
| Negated ("no production experience with X") | **no** | explicit denial |
| Aspirational ("currently learning X") | **no** | not yet held |
| Inside a pasted job description | **no** | employer's requirement, not candidate's claim |
| **Object of analysis** ("analysed **Android** malware") | **the object: no — the activity: yes** | see below |

### Object of analysis

`"Analysed Android malware samples"` does **not** evidence Android development.
It **does** evidence malware analysis. The parser therefore *redirects* rather
than discarding: the object mention is dropped and the activity skill recorded
in its place, logged as `MENTION_REDIRECTED`.

When labelling, apply the same rule: credit the capability the sentence
demonstrates, not the technology it was pointed at.

## What counts as experience

| Case | Decision |
|---|---|
| Paid employment with dates | **yes** |
| Internship with dates | **yes** |
| Freelance/contract with dates | **yes** |
| Teaching assistantship | **decide and record here** — currently NO |
| Academic project | **no** |
| Stated professional total, no dates ("10 years of experience") | yes, flagged `EXPERIENCE_FROM_TEXT` |
| Project duration ("built a system over 3 years") | **no** — not a career total |
| Nothing to go on | `null` — never estimate |

Employment intervals are authoritative and always win over a prose claim.
Overlapping roles are merged, so concurrent jobs are not double-counted.

## What counts as education

| Case | Decision |
|---|---|
| Degree (Bachelors and above) | **yes** |
| Diploma | **yes** |
| Class X / XII / secondary school | **no** |
| Online course or MOOC | **no** — a certification, not education |
| Ongoing degree | **yes**, with its expected year |

## Precision over recall

A missing skill is recoverable — a recruiter still reads the resume. A
**fabricated** skill is not, because nothing downstream can tell it was wrong.
If you would not defend the skill to the candidate, leave it out.
