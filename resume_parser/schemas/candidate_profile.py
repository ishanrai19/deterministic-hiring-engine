"""Typed contract between the Resume Screening Agent and the Skill Matching Agent."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.1"
SourceType = Literal["skills_section", "experience", "project", "certification"]
SOURCE_ORDER: List[str] = ["skills_section", "experience", "project", "certification"]


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def _ordered(self):
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox must satisfy x0 <= x1 and y0 <= y1")
        return self


class SkillLocation(BaseModel):
    source: SourceType
    page: int = Field(ge=1)
    bbox: Optional[BoundingBox] = None
    raw_text: Optional[str] = None


class Skill(BaseModel):
    skill_id: str
    skill_name: str
    raw_text: str
    sources: List[SourceType]
    evidence_count: int = Field(ge=0)
    locations: List[SkillLocation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self):
        if self.evidence_count != len(set(self.sources)):
            raise ValueError("evidence_count must equal the number of distinct sources")
        if self.locations and not {l.source for l in self.locations}.issubset(set(self.sources)):
            raise ValueError("every location source must appear in sources")
        return self


class Education(BaseModel):
    degree: Optional[str] = None
    field: Optional[str] = None
    institution: Optional[str] = None
    year: Optional[int] = None


class WorkHistory(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    # Verbatim summary of what the role involved, taken from the resume's own
    # bullets. Never paraphrased or generated - the Interview Agent needs the
    # candidate's wording, and a rewritten description cannot be cited back.
    description: Optional[str] = None


class Project(BaseModel):
    """A project as the candidate presented it.

    Kept separate from skill evidence on purpose: `skills` answers "what can
    this person do", `projects` answers "what did they build". The Interview
    Agent needs the second to ask specific questions.
    """

    title: str
    description: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class CandidateProfile(BaseModel):
    candidate_id: str
    source_resume_path: str
    education: List[Education] = Field(default_factory=list)
    experience_years: Optional[float] = Field(default=None, ge=0)
    work_history: List[WorkHistory] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    skills: List[Skill] = Field(default_factory=list)
    parsing_status: Literal["success", "partial", "failed"]
    schema_version: str = SCHEMA_VERSION
    parsing_timestamp: str

    @model_validator(mode="after")
    def _unique_skills(self):
        ids = [s.skill_id for s in self.skills]
        if len(ids) != len(set(ids)):
            raise ValueError("skills must be unique by skill_id")
        return self
