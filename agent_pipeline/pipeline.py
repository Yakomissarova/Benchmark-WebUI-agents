from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langgraph.graph import StateGraph, END

from .agents import SiteBuilderAgent, SiteReviewerAgent, SiteRepairAgent
from .executors import SiteApplier, SiteTester, SiteDockerPackager
from .llm_adapter import LLMAdapter
from .models import RepairDirective
from .state import SiteState


class SitePipeline:
    def __init__(
        self,
        llm: Any,
        *,
        base_workdir: Optional[str] = None,
        max_iterations: int = 2,
        enable_docker: bool = False,
        docker_tag_prefix: str = "bench/site",
    ):
        self.base_workdir = base_workdir
        self.max_iterations = max_iterations
        self.enable_docker = enable_docker
        self.docker_tag_prefix = docker_tag_prefix

        adapter = LLMAdapter(llm)
        self.builder = SiteBuilderAgent(adapter)
        self.reviewer = SiteReviewerAgent(adapter)
        self.repairer = SiteRepairAgent(adapter)

        self.applier = SiteApplier()
        self.tester = SiteTester()
        self.packager = SiteDockerPackager()

        self._graph = self._build_graph()

    # --------- routing (conditional transitions) ----------
    def _route_after_review(self, state: SiteState) -> str:
        assert state.review is not None
        if state.review.ok:
            return "materialize"
        if state.iteration >= state.max_iterations:
            return "fail"
        return "repair"

    def _route_after_test(self, state: SiteState) -> str:
        assert state.tests is not None
        if state.tests.ok:
            return "docker" if self.enable_docker else "done"
        if state.iteration >= state.max_iterations:
            return "done"
        return "repair"

    # --------- nodes (graph steps) ----------
    async def _n_init_workdir(self, state: SiteState) -> SiteState:
        state.max_iterations = self.max_iterations
        if state.workdir is None:
            state.workdir = tempfile.mkdtemp(prefix=f"{state.site_id}_", dir=self.base_workdir)
        state.traces.append(f"workdir={state.workdir}")
        return state

    async def _n_build(self, state: SiteState) -> SiteState:
        state.patchset = await self.builder.build_patchset(state)
        state.traces.append(f"built_artifacts={len(state.patchset.artifacts)}")
        return state

    async def _n_review(self, state: SiteState) -> SiteState:
        state.review = await self.reviewer.review(state)
        state.traces.append(f"review_ok={state.review.ok}, score={state.review.score}")
        return state

    async def _n_maybe_repair_plan(self, state: SiteState) -> SiteState:
        if state.review and (not state.review.ok) and state.iteration < state.max_iterations:
            state.repair = await self.reviewer.repair_directive(state)
            state.traces.append("repair_directive_ready")
        return state

    async def _n_repair(self, state: SiteState) -> SiteState:
        state.iteration += 1
        state.patchset = await self.repairer.apply_repairs(state)
        state.traces.append(f"repaired_iter={state.iteration}")
        return state

    async def _n_materialize(self, state: SiteState) -> SiteState:
        assert state.workdir is not None
        self.applier.materialize(state)
        state.traces.append("materialized")
        return state

    async def _n_test(self, state: SiteState) -> SiteState:
        assert state.workdir is not None
        root = Path(state.workdir)
        state.tests = self.tester.test(root)
        state.traces.append(f"tests_ok={state.tests.ok}")

        if not state.tests.ok:
            state.repair = RepairDirective(
                summary="Fix build/test failures",
                todo=[
                    "Inspect logs and fix missing imports/files/scripts",
                    "Ensure Vite build works and smoke test passes",
                ],
                required_changes=[state.tests.logs[-4000:]],
            )
            state.traces.append("repair_from_test_logs_ready")

        return state

    async def _n_docker(self, state: SiteState) -> SiteState:
        assert state.workdir is not None
        root = Path(state.workdir)
        tag = f"{self.docker_tag_prefix}:{state.site_id}".lower().replace(" ", "-")
        state.docker = self.packager.build_image(root, image_tag=tag)
        state.traces.append(f"docker_ok={state.docker.ok}, tag={tag}")
        return state

    # --------- graph definition ----------
    def _build_graph(self):
        g = StateGraph(SiteState)

        g.add_node("init_workdir", self._n_init_workdir)
        g.add_node("build", self._n_build)
        g.add_node("review", self._n_review)
        g.add_node("maybe_repair_plan", self._n_maybe_repair_plan)
        g.add_node("repair", self._n_repair)
        g.add_node("materialize", self._n_materialize)
        g.add_node("test", self._n_test)
        g.add_node("docker", self._n_docker)

        g.set_entry_point("init_workdir")
        g.add_edge("init_workdir", "build")
        g.add_edge("build", "review")
        g.add_edge("review", "maybe_repair_plan")

        g.add_conditional_edges(
            "maybe_repair_plan",
            self._route_after_review,
            {"repair": "repair", "materialize": "materialize", "fail": END},
        )

        g.add_edge("repair", "review")
        g.add_edge("materialize", "test")

        g.add_conditional_edges(
            "test",
            self._route_after_test,
            {"repair": "repair", "docker": "docker", "done": END},
        )

        g.add_edge("docker", END)

        return g.compile()

    # --------- public API ----------
    async def process_one(self, site_id: str, raw_templates: Dict[str, str]) -> SiteState:
        state = SiteState(site_id=site_id, raw_templates=raw_templates)
        return await self._graph.ainvoke(state)

    async def process_many(
        self, items: List[Tuple[str, Dict[str, str]]], concurrency: int = 4
    ) -> List[SiteState]:
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(site_id: str, tpls: Dict[str, str]) -> SiteState:
            async with sem:
                return await self.process_one(site_id, tpls)

        return await asyncio.gather(*[_run_one(sid, t) for sid, t in items])
