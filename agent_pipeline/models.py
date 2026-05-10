from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class CodeArtifact(BaseModel):
    path: str
    content: str


class BuildPlan(BaseModel):
    entry_html: str = Field(..., description="Path to the entry HTML file.")
    framework: str = Field("vite-vanilla", description="Target framework/build system identifier.")
    notes: Optional[str] = None


class PatchSet(BaseModel):
    plan: BuildPlan
    artifacts: List[CodeArtifact] = Field(default_factory=list)
    commands: List[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    severity: str = Field(..., description="low|med|high")
    file: Optional[str] = None
    message: str
    suggestion: Optional[str] = None


class ReviewReport(BaseModel):
    ok: bool
    score: int = Field(..., ge=0, le=100)
    findings: List[ReviewFinding] = Field(default_factory=list)


class RepairDirective(BaseModel):
    summary: str
    todo: List[str] = Field(default_factory=list)
    required_changes: List[str] = Field(default_factory=list)


class TestReport(BaseModel):
    ok: bool
    checks: List[str] = Field(default_factory=list)
    logs: str = ""


class DockerReport(BaseModel):
    ok: bool
    image_tag: Optional[str] = None
    build_logs: str = ""
