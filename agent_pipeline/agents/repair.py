from __future__ import annotations

from ..llm_adapter import LLMAdapter
from ..models import PatchSet
from ..state import SiteState


class SiteRepairAgent:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def apply_repairs(self, state: SiteState) -> PatchSet:
        assert state.patchset is not None
        assert state.repair is not None

        system = (
            "You are a senior engineer fixing a Vite project patchset.\n"
            "Return ONLY JSON for PatchSet.\n"
            "Apply RepairDirective with minimal changes.\n"
        )
        user = (
            f"Current PatchSet: {state.patchset.model_dump()}\n"
            f"RepairDirective: {state.repair.model_dump()}\n"
            "Return an updated PatchSet JSON.\n"
        )
        return await self.llm.ajson(system, user, PatchSet)
