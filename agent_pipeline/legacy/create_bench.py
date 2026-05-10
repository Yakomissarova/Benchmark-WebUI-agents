"""
Self-reflective LangGraph pipeline:
raw HTML templates -> agent "revives" + links (JS/CSS/build) -> reviewer -> tests -> docker packaging
Designed for async batch processing.

Deps (minimal):
  pip install langgraph langchain pydantic

LLM wiring:
  - Provide any LangChain chat model (e.g., ChatOpenAI) via SitePipeline(llm=...).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END


# ---------------------------- Models (structured I/O) ----------------------------

class CodeArtifact(BaseModel):
    path: str
    content: str

class BuildPlan(BaseModel):
    app_name: str = "site-app"
    entry_html: str = "index.html"
    framework: Literal["vanilla_vite"] = "vanilla_vite"
    notes: List[str] = Field(default_factory=list)

class PatchSet(BaseModel):
    plan: BuildPlan
    artifacts: List[CodeArtifact]
    commands: List[str] = Field(default_factory=list)

class ReviewFinding(BaseModel):
    severity: Literal["low", "medium", "high", "blocker"]
    message: str
    suggestion: Optional[str] = None

class ReviewReport(BaseModel):
    ok: bool
    score: int = Field(ge=0, le=100)
    findings: List[ReviewFinding] = Field(default_factory=list)

class RepairDirective(BaseModel):
    summary: str
    todo: List[str] = Field(default_factory=list)
    required_changes: List[str] = Field(default_factory=list)

class TestReport(BaseModel):
    ok: bool
    logs: str = ""
    checks: Dict[str, bool] = Field(default_factory=dict)

class DockerReport(BaseModel):
    ok: bool
    image_tag: str
    build_logs: str = ""


# ---------------------------- State ----------------------------

@dataclass
class SiteState:
    site_id: str
    raw_templates: Dict[str, str]  # filename -> html string
    workdir: Optional[str] = None

    patchset: Optional[PatchSet] = None
    review: Optional[ReviewReport] = None
    repair: Optional[RepairDirective] = None
    tests: Optional[TestReport] = None
    docker: Optional[DockerReport] = None

    iteration: int = 0
    max_iterations: int = 3

    # optional debug
    traces: List[str] = field(default_factory=list)


# ---------------------------- Utilities ----------------------------

def _safe_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def _run(cmd: List[str], cwd: Path, timeout_s: int = 900) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env={**os.environ, "CI": "1"},
    )
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, out

def _guess_entry_html(raw_templates: Dict[str, str]) -> str:
    if "index.html" in raw_templates:
        return "index.html"
    for k in raw_templates:
        if k.lower().endswith(".html"):
            return k
    return "index.html"

def _normalize_rel_paths(html: str) -> str:
    html = re.sub(r'href="\/', 'href="', html)
    html = re.sub(r'src="\/', 'src="', html)
    return html

def _default_package_json(app_name: str) -> str:
    pkg = {
        "name": app_name,
        "private": True,
        "type": "module",
        "scripts": {
            "dev": "vite --host 0.0.0.0 --port 5173",
            "build": "vite build",
            "preview": "vite preview --host 0.0.0.0 --port 4173",
            "test:smoke": "node ./scripts/smoke.mjs",
        },
        "devDependencies": {
            "vite": "^5.0.0"
        }
    }
    return json.dumps(pkg, indent=2)

def _default_vite_config() -> str:
    return """import { defineConfig } from 'vite';

export default defineConfig({
  server: { strictPort: true },
  build: { outDir: 'dist' }
});
"""

def _default_smoke_script(entry_html: str) -> str:
    return f"""import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const entry = path.join(root, '{entry_html}');
if (!fs.existsSync(entry)) {{
  console.error('Missing entry html:', entry);
  process.exit(1);
}}
const html = fs.readFileSync(entry, 'utf-8');
const hasApp = html.includes('id="app"') || html.includes("data-app");
const hasModule = html.includes('type="module"') || html.includes('src="/src/main.js"') || html.includes('src="src/main.js"');
if (!hasApp) console.warn('Smoke: no #app or data-app found (ok if static).');
if (!hasModule) console.warn('Smoke: no module script detected (will rely on injected main.js).');
console.log('Smoke OK');
"""

def _dockerfile_nginx() -> str:
    return """FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN if [ -f package-lock.json ]; then npm ci; elif [ -f pnpm-lock.yaml ]; then corepack enable && pnpm i --frozen-lockfile; elif [ -f yarn.lock ]; then corepack enable && yarn --frozen-lockfile; else npm i; fi
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
"""


# ---------------------------- LLM Adapter ----------------------------

class LLMAdapter:
    """
    Wrap any LangChain chat model:
      - must expose `ainvoke(messages)` returning AIMessage/content
    """

    def __init__(self, chat_model: Any):
        self.chat_model = chat_model

    async def ajson(self, system: str, user: str, schema: Any) -> Any:
        # Works with most LC chat models; expects JSON in response.
        messages = [
            ("system", system),
            ("user", user),
        ]
        resp = await self.chat_model.ainvoke(messages)
        content = getattr(resp, "content", resp)

        # extract first JSON object/array
        m = re.search(r"(\{.*\}|\[.*\])", content, flags=re.S)
        if not m:
            raise ValueError(f"LLM did not return JSON. Got:\n{content}")
        data = json.loads(m.group(1))
        return schema.model_validate(data)


# ---------------------------- Agents (OOP) ----------------------------

class SiteBuilderAgent:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def build_patchset(self, state: SiteState) -> PatchSet:
        entry = _guess_entry_html(state.raw_templates)

        system = (
            "You are a senior web engineer agent.\n"
            "Input: raw HTML templates only (no JS/CSS/build).\n"
            "Goal: 'revive' the site into a runnable Vite vanilla app:\n"
            " - create /src/main.js to attach basic interactivity and safe bindings\n"
            " - ensure an app mount point (#app or data-app) exists (inject if needed)\n"
            " - wire navigation, buttons, forms with unobtrusive JS using data-* hooks\n"
            " - keep original HTML structure; avoid breaking layout\n"
            " - output a PatchSet JSON: plan + artifacts + commands.\n"
            "Constraints:\n"
            " - Be conservative: no external APIs required\n"
            " - Provide minimal CSS only if needed\n"
            " - Must include package.json, vite.config.js, src/main.js\n"
            " - Include a small scripts/smoke.mjs\n"
            " - Use relative asset paths\n"
            "Return ONLY valid JSON."
        )

        # Provide all templates, but keep prompt bounded
        templ_dump = []
        for name, html in state.raw_templates.items():
            html_n = _normalize_rel_paths(html)
            if len(html_n) > 40_000:
                html_n = html_n[:40_000] + "\n<!-- TRUNCATED -->\n"
            templ_dump.append(f"---FILE:{name}---\n{html_n}\n")
        templates_blob = "\n".join(templ_dump)

        user = (
            f"Site ID: {state.site_id}\n"
            f"Entry HTML guess: {entry}\n"
            f"Templates:\n{templates_blob}\n\n"
            "Produce PatchSet JSON now."
        )

        patchset = await self.llm.ajson(system, user, PatchSet)

        # Hard-guard: ensure critical files exist
        required = {"package.json", "vite.config.js", "src/main.js", "scripts/smoke.mjs"}
        missing = [p for p in required if p not in {a.path for a in patchset.artifacts}]
        if missing:
            # Inject defaults if LLM forgot
            artifacts = list(patchset.artifacts)
            if "package.json" in missing:
                artifacts.append(CodeArtifact(path="package.json", content=_default_package_json(patchset.plan.app_name)))
            if "vite.config.js" in missing:
                artifacts.append(CodeArtifact(path="vite.config.js", content=_default_vite_config()))
            if "scripts/smoke.mjs" in missing:
                artifacts.append(CodeArtifact(path="scripts/smoke.mjs", content=_default_smoke_script(patchset.plan.entry_html)))
            if "src/main.js" in missing:
                artifacts.append(CodeArtifact(path="src/main.js", content="console.log('main.js ready');"))
            patchset = PatchSet(plan=patchset.plan, artifacts=artifacts, commands=patchset.commands)

        return patchset


class SiteReviewerAgent:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def review(self, state: SiteState) -> ReviewReport:
        system = (
            "You are a strict reviewer agent for a generated web project.\n"
            "Evaluate:\n"
            " - correctness of Vite vanilla wiring\n"
            " - JS attaches to DOM safely, avoids fragile selectors, uses data-* where possible\n"
            " - forms/buttons have predictable behavior\n"
            " - build/test scripts exist\n"
            " - no broken imports, paths, or obvious runtime errors\n"
            "Return ReviewReport JSON only."
        )

        patch = state.patchset
        assert patch is not None

        # Provide a compact listing + key file contents
        def pick(paths: List[str]) -> str:
            by = {a.path: a.content for a in patch.artifacts}
            out = []
            for p in paths:
                if p in by:
                    c = by[p]
                    if len(c) > 20_000:
                        c = c[:20_000] + "\n/* TRUNCATED */\n"
                    out.append(f"---{p}---\n{c}\n")
            return "\n".join(out)

        user = (
            f"Site ID: {state.site_id}\n"
            f"Iteration: {state.iteration}\n"
            f"Plan: {patch.plan.model_dump()}\n"
            f"Commands: {patch.commands}\n"
            f"Key files:\n"
            f"{pick([patch.plan.entry_html, 'package.json', 'vite.config.js', 'src/main.js', 'src/styles.css', 'scripts/smoke.mjs'])}\n"
            "Return ReviewReport JSON."
        )

        return await self.llm.ajson(system, user, ReviewReport)

    async def repair_directive(self, state: SiteState) -> RepairDirective:
        system = (
            "You are a repair planner.\n"
            "Given the review findings, produce concrete changes required to pass.\n"
            "Return RepairDirective JSON only."
        )
        assert state.review is not None
        patch = state.patchset
        assert patch is not None

        user = (
            f"Site ID: {state.site_id}\n"
            f"Plan: {patch.plan.model_dump()}\n"
            f"Review: {state.review.model_dump()}\n"
            "Return RepairDirective JSON."
        )
        return await self.llm.ajson(system, user, RepairDirective)


class SiteRepairAgent:
    def __init__(self, llm: LLMAdapter):
        self.llm = llm

    async def apply_repairs(self, state: SiteState) -> PatchSet:
        system = (
            "You are a code-fixing agent.\n"
            "Input: current PatchSet and RepairDirective.\n"
            "Output: new PatchSet JSON with corrected artifacts.\n"
            "Constraints:\n"
            " - keep changes minimal\n"
            " - ensure Vite build works\n"
            "Return ONLY JSON."
        )

        assert state.patchset is not None
        assert state.repair is not None

        # Provide full artifacts list but truncate huge ones
        artifacts_dump = []
        for a in state.patchset.artifacts:
            c = a.content
            if len(c) > 30_000:
                c = c[:30_000] + "\n/* TRUNCATED */\n"
            artifacts_dump.append({"path": a.path, "content": c})

        user = (
            f"Site ID: {state.site_id}\n"
            f"Current plan: {state.patchset.plan.model_dump()}\n"
            f"Repair directive: {state.repair.model_dump()}\n"
            f"Current artifacts: {artifacts_dump}\n"
            "Return updated PatchSet JSON."
        )

        return await self.llm.ajson(system, user, PatchSet)


# ---------------------------- Executors (local) ----------------------------

class SiteWorkspace:
    def __init__(self, base_dir: Optional[str] = None):
        self._tmp = None
        self.base_dir = base_dir

    def __enter__(self) -> Path:
        self._tmp = tempfile.mkdtemp(prefix="sitepipe_", dir=self.base_dir)
        return Path(self._tmp)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmp and Path(self._tmp).exists():
            shutil.rmtree(self._tmp, ignore_errors=True)

class SiteApplier:
    def materialize(self, state: SiteState) -> Path:
        assert state.workdir is not None
        root = Path(state.workdir)

        # Write raw templates first
        for name, html in state.raw_templates.items():
            name = name.strip().lstrip("/").replace("\\", "/")
            _safe_write(root / name, _normalize_rel_paths(html))

        # Apply generated artifacts after
        assert state.patchset is not None
        for a in state.patchset.artifacts:
            p = a.path.strip().lstrip("/").replace("\\", "/")
            _safe_write(root / p, a.content)

        return root

class SiteTester:
    def __init__(self, npm_client: Literal["npm"] = "npm"):
        self.npm = npm_client

    def test(self, root: Path) -> TestReport:
        checks: Dict[str, bool] = {}

        rc, out = _run([self.npm, "--version"], cwd=root, timeout_s=60)
        if rc != 0:
            return TestReport(ok=False, logs="npm missing?\n" + out, checks={"npm": False})
        checks["npm"] = True

        # install
        rc, out_i = _run([self.npm, "install"], cwd=root, timeout_s=900)
        ok_install = (rc == 0)
        checks["install"] = ok_install
        if not ok_install:
            return TestReport(ok=False, logs=out_i, checks=checks)

        # smoke
        rc, out_s = _run([self.npm, "run", "test:smoke"], cwd=root, timeout_s=180)
        ok_smoke = (rc == 0)
        checks["smoke"] = ok_smoke
        if not ok_smoke:
            return TestReport(ok=False, logs=out_s, checks=checks)

        # build
        rc, out_b = _run([self.npm, "run", "build"], cwd=root, timeout_s=900)
        ok_build = (rc == 0)
        checks["build"] = ok_build
        logs = out_i + "\n" + out_s + "\n" + out_b
        return TestReport(ok=ok_build and ok_smoke and ok_install, logs=logs, checks=checks)

class SiteDockerPackager:
    def __init__(self, docker_bin: str = "docker"):
        self.docker = docker_bin

    def ensure_dockerfile(self, root: Path) -> None:
        df = root / "Dockerfile"
        if not df.exists():
            _safe_write(df, _dockerfile_nginx())
        if not (root / ".dockerignore").exists():
            _safe_write(
                root / ".dockerignore",
                "node_modules\n.dist\n.vite\n.DS_Store\n__pycache__\nsitepipe_*\n",
            )

    def build_image(self, root: Path, image_tag: str) -> DockerReport:
        rc, out = _run([self.docker, "version"], cwd=root, timeout_s=60)
        if rc != 0:
            return DockerReport(ok=False, image_tag=image_tag, build_logs="Docker not available:\n" + out)

        self.ensure_dockerfile(root)
        rc, out_b = _run([self.docker, "build", "-t", image_tag, "."], cwd=root, timeout_s=1800)
        return DockerReport(ok=(rc == 0), image_tag=image_tag, build_logs=out_b)


# ---------------------------- Main Pipeline (LangGraph) ----------------------------

class SitePipeline:
    def __init__(
        self,
        llm_chat_model: Any,
        *,
        max_iterations: int = 3,
        base_workdir: Optional[str] = None,
        enable_docker: bool = True,
        docker_tag_prefix: str = "sitepipe",
        concurrency: int = 4,
    ):
        self.llm = LLMAdapter(llm_chat_model)
        self.builder = SiteBuilderAgent(self.llm)
        self.reviewer = SiteReviewerAgent(self.llm)
        self.repairer = SiteRepairAgent(self.llm)

        self.applier = SiteApplier()
        self.tester = SiteTester()
        self.packager = SiteDockerPackager()

        self.max_iterations = max_iterations
        self.base_workdir = base_workdir
        self.enable_docker = enable_docker
        self.docker_tag_prefix = docker_tag_prefix
        self.concurrency = concurrency

        self._graph = self._build_graph()

    # ---- Graph construction ----

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
            {
                "repair": "repair",
                "materialize": "materialize",
                "fail": END,
            },
        )

        g.add_edge("repair", "review")          # self-reflection loop
        g.add_edge("materialize", "test")

        g.add_conditional_edges(
            "test",
            self._route_after_test,
            {
                "repair": "repair",
                "docker": "docker",
                "done": END,
            },
        )

        g.add_edge("docker", END)

        return g.compile()

    # ---- Routing ----

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

    # ---- Nodes ----

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
        state.traces.append(f"review_ok={state.review.ok},score={state.review.score}")
        return state

    async def _n_maybe_repair_plan(self, state: SiteState) -> SiteState:
        if state.review and (not state.review.ok) and state.iteration < state.max_iterations:
            state.repair = await self.reviewer.repair_directive(state)
            state.traces.append("repair_directive_ready")
        return state

    async def _n_repair(self, state: SiteState) -> SiteState:
        state.iteration += 1
        state.patchset = await self.repairer.apply_repairs(state)
        state.traces.append(f"repaired_iteration={state.iteration}")
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
            # convert failing tests into repair directive if reviewer step isn't enough
            state.repair = RepairDirective(
                summary="Fix build/test failures",
                todo=["Make npm install/build pass", "Fix path/import/runtime issues reported in logs"],
                required_changes=[state.tests.logs[-4000:]],
            )
        return state

    async def _n_docker(self, state: SiteState) -> SiteState:
        assert state.workdir is not None
        root = Path(state.workdir)
        tag = f"{self.docker_tag_prefix}:{state.site_id}".lower().replace(" ", "-")
        state.docker = self.packager.build_image(root, image_tag=tag)
        state.traces.append(f"docker_ok={state.docker.ok}")
        return state

    # ---- Public API ----

    async def process_one(self, site_id: str, raw_templates: Dict[str, str]) -> SiteState:
        state = SiteState(site_id=site_id, raw_templates=raw_templates, max_iterations=self.max_iterations)
        final_state: SiteState = await self._graph.ainvoke(state)
        return final_state

    async def process_many(self, items: List[Tuple[str, Dict[str, str]]]) -> List[SiteState]:
        sem = asyncio.Semaphore(self.concurrency)

        async def _wrap(site_id: str, raw: Dict[str, str]) -> SiteState:
            async with sem:
                st = await self.process_one(site_id, raw)
                return st

        return await asyncio.gather(*[_wrap(sid, raw) for sid, raw in items])


# ---------------------------- Example usage ----------------------------

async def _example():
    # Example: you supply a real LC chat model here.
    #
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    #
    # pipeline = SitePipeline(llm, concurrency=8, enable_docker=True)
    #
    # templates = {"index.html": "<!doctype html><html><body><h1>Hello</h1></body></html>"}
    # result = await pipeline.process_one("demo", templates)
    # print("OK:", result.tests.ok if result.tests else None, "Docker:", result.docker.ok if result.docker else None)
    # print("Workdir:", result.workdir)
    # print("Traces:", result.traces)
    pass


if __name__ == "__main__":
    asyncio.run(_example())

