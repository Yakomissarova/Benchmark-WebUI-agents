from __future__ import annotations

from ..llm_adapter import LLMAdapter
from ..models import ReviewReport, RepairDirective
from ..state import SiteState


class SiteReviewerAgent:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def review(self, state: SiteState) -> ReviewReport:
        assert state.patchset is not None

        system = (
            "You are a strict code reviewer.\n"
            "Return ONLY JSON for ReviewReport.\n"
            "Check that Vite project wiring makes sense, scripts exist, entry html is handled.\n"
        )

        files = {a.path: a.content[:3000] for a in state.patchset.artifacts}
        user = (
            f"Patch plan: {state.patchset.plan.model_dump()}\n"
            f"Commands: {state.patchset.commands}\n"
            f"Key artifacts (truncated): {files}\n"
            "Return ReviewReport JSON.\n"
        )
        return await self.llm.ajson(system, user, ReviewReport)

    async def repair_directive(self, state: SiteState) -> RepairDirective:
        assert state.review is not None
        assert state.patchset is not None

        system = (
            "You produce concrete repair instructions.\n"
            "Return ONLY JSON for RepairDirective.\n"
            "Make changes minimal and actionable.\n"
        )

        user = (
            f"Review findings: {state.review.model_dump()}\n"
            f"Current plan: {state.patchset.plan.model_dump()}\n"
            "Produce RepairDirective JSON.\n"
        )
        return await self.llm.ajson(system, user, RepairDirective)
