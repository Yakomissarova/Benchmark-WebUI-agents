from __future__ import annotations

from typing import Dict

from ..llm_adapter import LLMAdapter
from ..models import PatchSet, CodeArtifact
from ..state import SiteState
from ..utils import (
    guess_entry_html,
    normalize_rel_paths,
    default_package_json,
    default_vite_config,
    default_smoke_script,
)


class SiteBuilderAgent:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def build_patchset(self, state: SiteState) -> PatchSet:
        entry_html = guess_entry_html(state.raw_templates)

        system = (
            "You are a senior frontend engineer.\n"
            "Return ONLY valid JSON for the requested schema.\n"
            "Goal: revive raw HTML templates into a working Vite vanilla project.\n"
            "You must generate at least:\n"
            "- package.json with scripts dev/build/test:smoke\n"
            "- vite.config.js (or .ts)\n"
            "- src/main.js (wires DOM)\n"
            "- scripts/smoke.mjs\n"
        )

        templates: Dict[str, str] = {}
        for k, v in state.raw_templates.items():
            templates[k] = normalize_rel_paths(v[:20000])  # trim huge pages

        user = (
            f"Site ID: {state.site_id}\n"
            f"Entry HTML: {entry_html}\n"
            "Templates (path -> html):\n"
            f"{templates}\n\n"
            "Produce a PatchSet JSON.\n"
            "Artifacts should include the generated project files.\n"
        )

        patchset = await self.llm.ajson(system, user, PatchSet)

        # Ensure critical files exist (keep same behavior as monolith hard-guards)
        paths = {a.path for a in patchset.artifacts}
        app_name = state.site_id.lower().replace(" ", "-")

        def add(path: str, content: str) -> None:
            patchset.artifacts.append(CodeArtifact(path=path, content=content))

        if "package.json" not in paths:
            add("package.json", default_package_json(app_name))
        if "vite.config.js" not in paths and "vite.config.ts" not in paths:
            add("vite.config.js", default_vite_config())
        if "scripts/smoke.mjs" not in paths:
            add("scripts/smoke.mjs", default_smoke_script(entry_html))

        return patchset
