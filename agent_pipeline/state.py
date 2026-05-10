from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List

from .models import PatchSet, ReviewReport, RepairDirective, TestReport, DockerReport


@dataclass
class SiteState:
    site_id: str
    raw_templates: Dict[str, str]

    workdir: Optional[str] = None

    patchset: Optional[PatchSet] = None
    review: Optional[ReviewReport] = None
    repair: Optional[RepairDirective] = None
    tests: Optional[TestReport] = None
    docker: Optional[DockerReport] = None

    iteration: int = 0
    max_iterations: int = 2

    traces: List[str] = field(default_factory=list)
