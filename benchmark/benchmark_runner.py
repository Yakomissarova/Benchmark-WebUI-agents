import argparse
import asyncio
import json
import re
import os
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from playwright.async_api import async_playwright, Page

from benchmark.base_agent import BaseAgent


def create_agent(agent_type: str, model: str | None = None) -> BaseAgent:
    """
    Create agent by name.

    Supported values:
    - openrouter
    - browser_use
    """
    agent_type = agent_type.lower()

    if agent_type == "openrouter":
        from benchmark.openrouter_agent import OpenRouterAgent
        return OpenRouterAgent(model=model)

    elif agent_type == "browser_use":
        from benchmark.browser_use_agent import BrowserUseAgent
        model_name = model or os.getenv("OPEN_ROUTER_MODEL")

        return BrowserUseAgent(
            name=f"BrowserUse ({model_name})",
            model=model_name
        )

    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Supported: openrouter, browser_use"
        )


class BenchmarkRunner:
    """
    Runs benchmark scenarios with a given agent.

    The runner is responsible for:
    - loading scenarios
    - preparing browser/page
    - attaching HAR
    - calling the agent
    - evaluating success
    - collecting metrics
    """

    def __init__(
        self,
        benchmarks_path: str,
        agent: BaseAgent,
        headless: bool = False,
        cdp_url: str = "http://localhost:9222",
    ):
        self.benchmarks_path = benchmarks_path
        self.agent = agent
        self.headless = headless
        self.cdp_url = cdp_url
        self.scenarios = self._load_scenarios()
        self.results = []

    def _load_scenarios(self) -> List[Dict]:
        with open(self.benchmarks_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _normalize_eval_values(self, eval_value):
        if isinstance(eval_value, (list, tuple)):
            return [str(value) for value in eval_value if value is not None]
        if eval_value is None:
            return []
        return [str(eval_value)]

    def _url_matches(self, url: str, eval_type: str, eval_value) -> bool:
        values = self._normalize_eval_values(eval_value)
        if not url or not values:
            return False

        if eval_type == "url_contains":
            return any(value in url for value in values)
        if eval_type == "url_pattern":
            return any(bool(re.search(value, url)) for value in values)
        return False

    async def _evaluate_success(self, page: Page, scenario: Dict) -> bool:
        """
        Check whether the scenario goal is achieved.
        """
        eval_type = scenario.get("eval_type", "url_contains")
        eval_value = scenario.get("eval_value", "")

        current_url = page.url
        current_content = await page.content()

        if eval_type == "url_contains":
            return self._url_matches(current_url, eval_type, eval_value)

        elif eval_type == "url_pattern":
            return self._url_matches(current_url, eval_type, eval_value)

        elif eval_type == "element_exists":
            try:
                elem = await page.query_selector(eval_value)
                return elem is not None
            except Exception:
                return False

        elif eval_type == "text_contains":
            import re
            page_text = await page.evaluate("() => document.body.innerText")

            def normalize(s: str) -> str:
                s = s.lower()
                s = re.sub(r'[^\w\s]', ' ', s)  # убираем пунктуацию
                s = re.sub(r'\s+', ' ', s)  # схлопываем пробелы
                return s.strip()

            return normalize(eval_value) in normalize(page_text)

        else:
            print(f"⚠ Unknown evaluation type: {eval_type}")
            return False

    def _evaluate_trajectory(self, scenario: Dict, agent_result: Dict) -> Dict:
        """
        Analyse the full trajectory to check:
        - visited_eval_url: agent reached the target URL at any step
        - stopped_correctly: agent issued "stop" exactly when on the target URL
        """
        eval_type = scenario.get("eval_type", "url_contains")
        eval_value = scenario.get("eval_value", "")

        # [FIX] visited/stopped only make sense for URL-based eval types
        if eval_type not in ("url_contains", "url_pattern"):
            return {
                "visited_eval_url": None,
                "stopped_correctly": None,
            }

        trajectory = agent_result.get("trajectory", [])

        def url_matches(url: str) -> bool:
            return self._url_matches(url, eval_type, eval_value)

        visited_eval_url = False
        stopped_correctly = False

        for step in trajectory:
            url_after = step.get("url_after", "")
            action = step.get("action")

            if url_matches(url_after):
                visited_eval_url = True

            if action == "stop" and url_matches(url_after):
                stopped_correctly = True

        # Also check the very first state (start URL) just in case
        states = agent_result.get("states", [])
        for state in states:
            if url_matches(state.get("url", "")):
                visited_eval_url = True

        return {
            "visited_eval_url": visited_eval_url,
            "stopped_correctly": stopped_correctly,
        }

    def _evaluate_checkpoints(self, scenario: Dict, agent_result: Dict) -> Dict:
        checkpoints = scenario.get("checkpoints", [])
        if not checkpoints:
            return {"checkpoints_total": 0, "checkpoints_reached": 0}

        all_urls = [s.get("url", "") for s in agent_result.get("states", [])]
        all_urls += [s.get("url_after", "") for s in agent_result.get("trajectory", [])]

        reached = 0
        for cp in checkpoints:
            if any(cp in url for url in all_urls):
                reached += 1
            else:
                break

        return {"checkpoints_total": len(checkpoints), "checkpoints_reached": reached}


    async def _prepare_page(self, playwright):
        """
        Prepare browser/page depending on agent type.

        browser_mode:
        - "playwright_page" -> normal Playwright browser
        - "shared_cdp" -> connect to already running Chrome via CDP
        """
        if self.agent.browser_mode == "shared_cdp":
            browser = await playwright.chromium.connect_over_cdp(self.cdp_url)

            # Reuse existing default context if present.
            # This is the simplest option for CDP-based agents such as browser-use.
            if browser.contexts:
                context = browser.contexts[0]
            else:
                context = await browser.new_context(viewport={"width": 1280, "height": 800})

            page = await context.new_page()

            # Close all old pages so the agent does not jump to stale tabs.
            for p in list(context.pages):
                if p != page and not p.is_closed():
                    try:
                        await p.close()
                    except Exception:
                        pass

            await page.bring_to_front()

            # We do not want to close the whole external Chrome from the benchmark.
            should_close_browser = False

        else:
            browser = await playwright.chromium.launch(headless=self.headless)
            context = await browser.new_context(viewport={"width": 1280, "height": 800})
            page = await context.new_page()
            should_close_browser = True

        return browser, context, page, should_close_browser

    async def _get_eval_page(self, context, fallback_page, agent_result):
        """
        In shared CDP mode the agent may continue on another tab/page.
        Try to find the page that best matches the agent final URL.
        """
        pages = [p for p in context.pages if not p.is_closed()]
        if not pages:
            return fallback_page

        target_url = agent_result.get("final_url")

        if target_url:
            for p in reversed(pages):
                if p.url == target_url:
                    return p

        # Fallback: last non-empty page
        non_blank_pages = [p for p in pages if p.url and p.url != "about:blank"]
        if non_blank_pages:
            return non_blank_pages[-1]

        return fallback_page

    async def run_scenario(self, scenario: Dict) -> Dict:
        """
        Run a single benchmark scenario.
        """
        scenario_id = scenario.get("id")
        scenario_name = scenario.get("name")
        har_path = scenario.get("har_path")
        start_url = scenario.get("start_url")
        goal = scenario.get("goal")

        # otherwise fall back to a fixed date so that "past" dates still render correctly.
        har_file = Path(har_path).absolute()
        fake_date = None
        if har_file.exists():
            try:
                with open(har_file, "r", encoding="utf-8") as f:
                    har_data = json.load(f)
                fake_date = (
                    har_data.get("log", {})
                    .get("pages", [{}])[0]
                    .get("startedDateTime")
                )
            except Exception:
                pass
        if not fake_date:
            fake_date = "2026-03-12T12:00:00Z"

        print(f"\n{'=' * 60}")
        print(f"Scenario: {scenario_name} (ID: {scenario_id})")
        print(f"{'=' * 60}")

        result = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "eval_type": scenario.get("eval_type", "url_contains"),
            "success": False,
            "visited_eval_url": False,
            "stopped_correctly": False,
            "steps_taken": 0,
            "final_url": "",
            "error": None,
            "trajectory": [],
            "states": [],
            "timing": {},
            "usage": {},
            "extra": {},
        }

        browser = None
        page = None
        should_close_browser = False

        try:
            async with async_playwright() as p:
                browser, context, page, should_close_browser = await self._prepare_page(p)

                # [CHANGE 1] Freeze Date to the HAR recording time so that
                # date-sensitive UI (calendars, countdowns) renders as recorded.
                # await context.add_init_script(f"""
                #     const fakeNow = new Date('{fake_date}').getTime();
                #     const NativeDate = window.Date;
                #     window.Date = class extends NativeDate {{
                #         constructor(...args) {{
                #             if (args.length === 0) return new NativeDate(fakeNow);
                #             return new NativeDate(...args);
                #         }}
                #         static now() {{ return fakeNow; }}
                #     }};
                # """)

                # Attach HAR before navigation.
                if har_file.exists():
                    await page.route_from_har(str(har_file), update=False)
                else:
                    raise FileNotFoundError(f"⚠ HAR file not found: {har_path}")

                print(f"Go to {start_url}")
                await page.goto(start_url, wait_until="commit")
                await page.wait_for_timeout(2000)

                # Extra runtime information for agents that need it.
                runtime = {
                    "scenario": scenario,
                    "browser_mode": self.agent.browser_mode,
                    "cdp_url": self.cdp_url if self.agent.browser_mode == "shared_cdp" else None,
                }

                # Run agent
                agent_result = await self.agent.act(page, goal, max_steps=20, runtime=runtime)

                # Required fields
                result["steps_taken"] = agent_result.get("steps_taken", 0)
                result["final_url"] = agent_result.get("final_url", page.url)
                result["error"] = agent_result.get("error")

                # Optional fields for future metrics
                result["trajectory"] = agent_result.get("trajectory", [])
                result["states"] = agent_result.get("states", [])
                result["timing"] = agent_result.get("timing", {})
                result["usage"] = agent_result.get("usage", {})
                result["extra"] = agent_result.get("extra", {})

                # [CHANGE 2] Trajectory-based metrics: visited target URL + correct stop
                traj_eval = self._evaluate_trajectory(scenario, agent_result)
                result["visited_eval_url"] = traj_eval["visited_eval_url"]
                result["stopped_correctly"] = traj_eval["stopped_correctly"]

                # Checkpoint progress
                cp_eval = self._evaluate_checkpoints(scenario, agent_result)
                result["checkpoints_total"] = cp_eval["checkpoints_total"]
                result["checkpoints_reached"] = cp_eval["checkpoints_reached"]

                # Runner remains the source of truth for final success.
                eval_page = await self._get_eval_page(context, page, agent_result)
                try:
                    await eval_page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass
                success = await self._evaluate_success(eval_page, scenario)

                # Save the actual evaluated page URL
                result["final_url"] = eval_page.url

                visited_success = bool(traj_eval.get("visited_eval_url"))

                if agent_result.get("success") and success:
                    result["success"] = True
                    print("Scenario successful")
                elif success:
                    result["success"] = True
                    print(f"Scenario successful (by {scenario.get('eval_type')})")
                elif visited_success:
                    result["success"] = True
                    print(f"Scenario successful (visited target URL during trajectory)")
                else:
                    print("Scenario not successful")
                    if not agent_result.get("success") and agent_result.get("error"):
                        print(f"   ⚠ Error: {agent_result['error']}")

        except Exception as e:
            result["error"] = str(e)
            print(f"⚠ Error occurred while running scenario: {e}")

        finally:
            # For CDP mode we should not close the whole external browser.
            try:
                if page is not None and not page.is_closed():
                    await page.close()
            except Exception:
                pass

            try:
                if browser is not None and should_close_browser:
                    await browser.close()
            except Exception:
                pass

        return result

    async def _run_scenario_with_retry(self, scenario, max_attempts=5):
        """
        Запускает один сценарий и автоматически повторяет его,
        если он упал из-за транзиентной ошибки сети / LLM.

        Не повторяет, если агент честно завершился (success=True),
        упал в agent_stuck или исчерпал max_steps — иначе метрики будут завышены.
        """
        transient_markers = (
            "timed out", "timeout",
            "connection", "temporarily unavailable",
            "rate limit", "429",
            "502", "503", "500",
            "networkerror", "remote disconnected",
            "apiconnection", "apitimeout",
            "read operation timed out",
        )
        delays = [5, 15, 30, 60, 120]
        last_result = None

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                sleep_for = delays[min(attempt - 2, len(delays) - 1)]
                print(f"↻ Retry scenario {scenario.get('id')} "
                      f"(attempt {attempt}/{max_attempts}) in {sleep_for}s ...")
                await asyncio.sleep(sleep_for)

            result = await self.run_scenario(scenario)
            last_result = result

            err = (result.get("error") or "").lower()
            is_transient = any(kw in err for kw in transient_markers)

            if result.get("success") or not is_transient:
                if attempt > 1:
                    result.setdefault("extra", {})["retried_attempts"] = attempt
                return result

            print(f"⚠ Scenario {scenario.get('id')} transient error "
                  f"on attempt {attempt}/{max_attempts}: {err!r}")

        last_result.setdefault("extra", {})["retried_attempts"] = max_attempts
        return last_result





    async def run_all(self) -> Dict:
        """
        Run all benchmark scenarios and collect metrics.
        """
        print(f"\n{'#' * 60}")
        print(f"# Agent: {self.agent.name}")
        print(f"# Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# Total Scenarios: {len(self.scenarios)}")
        print(f"{'#' * 60}\n")

        self.results = []
        for scenario in self.scenarios:
            result = await self._run_scenario_with_retry(scenario)
            self.results.append(result)

        # Basic benchmark metrics
        successful = sum(1 for r in self.results if r["success"])
        total = len(self.results)
        success_rate = (successful / total * 100) if total > 0 else 0
        avg_steps = sum(r["steps_taken"] for r in self.results) / total if total > 0 else 0

        # Optional timing metric if agents provide timing
        durations = []
        for r in self.results:
            duration = r.get("timing", {}).get("duration_sec")
            if duration is not None:
                durations.append(duration)

        avg_duration = sum(durations) / len(durations) if durations else 0

        # Optional usage metric if agents provide token counts
        total_tokens_used = 0
        for r in self.results:
            total_tokens_used += r.get("usage", {}).get("total_tokens", 0) or 0

        # MORE METRICS

        total_useless_actions = 0
        total_actions = 0
        total_error_steps = 0

        for r in self.results:
            trajectory = r.get("trajectory", [])

            for step in trajectory:
                action = step.get("action")
                changed = step.get("changed_state")
                error = step.get("error")

                total_actions += 1

                # Count error steps
                if error:
                    total_error_steps += 1

                # Useless action heuristic
                if (
                    not changed
                    and action not in ["type", "stop"]
                ):
                    total_useless_actions += 1

        useless_ratio = (
            total_useless_actions / total_actions
            if total_actions > 0 else 0
        )

        error_ratio = (
            total_error_steps / total_actions
            if total_actions > 0 else 0
        )

        # Avg steps for successful only
        successful_steps = [
            r["steps_taken"] for r in self.results if r["success"]
        ]
        avg_steps_success = (
            sum(successful_steps) / len(successful_steps)
            if successful_steps else 0
        )

        # [CHANGE 2] Trajectory-based metrics aggregation
        visited_count = sum(1 for r in self.results if r.get("visited_eval_url"))
        stopped_correctly_count = sum(1 for r in self.results if r.get("stopped_correctly"))

        metrics = {
            "agent_name": self.agent.name,
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate_percent": round(success_rate, 2),

            "avg_steps": round(avg_steps, 2),
            "avg_steps_success_only": round(avg_steps_success, 2),

            "avg_duration_sec": round(avg_duration, 2),
            "total_tokens_used": total_tokens_used,

            # more
            "total_actions": total_actions,
            "useless_actions": total_useless_actions,
            "useless_actions_ratio": round(useless_ratio, 3),
            "error_steps": total_error_steps,
            "error_ratio": round(error_ratio, 3),

            # [CHANGE 2] trajectory-based
            "visited_eval_url_count": visited_count,
            "visited_eval_url_rate": round(visited_count / total * 100, 2) if total > 0 else 0,
            "stopped_correctly_count": stopped_correctly_count,
            "stopped_correctly_rate": round(stopped_correctly_count / total * 100, 2) if total > 0 else 0,
            # checkpoint progress
            "avg_checkpoint_progress": round(
                sum(
                    (r.get("checkpoints_reached", 0) / r["checkpoints_total"] * 100)
                    for r in self.results if r.get("checkpoints_total", 0) > 0
                ) / max(sum(1 for r in self.results if r.get("checkpoints_total", 0) > 0), 1), 2
            ),

            "results": self.results,
        }

        return metrics

    def print_report(self, metrics: Dict):
        """Print benchmark summary to console."""
        try:
            print(f"\n{'#' * 60}")
            print(f"# Final Report")
            print(f"{'#' * 60}")
            print(f"Agent:          {metrics['agent_name']}")
            print(f"Timestamp:      {metrics['timestamp']}")
            print(f"Total Scenarios:{metrics['total_scenarios']}")
            print(f"✅ Successful:  {metrics['successful']}")
            print(f"❌ Failed:      {metrics['failed']}")
            print(f"")
            print(f"--- Quality Metrics ---")
            print(f"   Success Rate (SR):              {metrics['success_rate_percent']}%")
            print(f"   Avg Steps per Scenario:         {metrics['avg_steps']}")
            if metrics['successful'] > 0:
                print(f"   Avg Steps (successful only):    {metrics['avg_steps_success_only']}")
            print(f"   Avg Duration per Scenario:      {metrics['avg_duration_sec']} sec")
            print(f"   Total Tokens Used:              {metrics['total_tokens_used']}")
            print(f"")
            print(f"--- Action Metrics ---")
            print(f"   Total Actions:                  {metrics['total_actions']}")
            if metrics['total_actions'] > 0:
                print(
                    f"   Useless Actions:                {metrics['useless_actions']} ({metrics['useless_actions_ratio'] * 100:.1f}%)")
                print(
                    f"   Error Steps:                    {metrics['error_steps']} ({metrics['error_ratio'] * 100:.1f}%)")
            print(f"")
            url_based = [r for r in metrics["results"] if r.get("eval_type") in ("url_contains", "url_pattern")]
            url_based_total = len(url_based)
            if url_based_total > 0:
                visited_count = metrics["visited_eval_url_count"]
                stopped_count = metrics["stopped_correctly_count"]
                print(f"--- Navigation Metrics (URL-based scenarios: {url_based_total}) ---")
                print(f"   Visited Target URL (any step):  {visited_count} ({metrics['visited_eval_url_rate']}%)")
                print(f"   Stopped Correctly on Target:    {stopped_count} ({metrics['stopped_correctly_rate']}%)")
                print(f"")
            cp_with_data = sum(1 for r in metrics["results"] if r.get("checkpoints_total", 0) > 0)
            if cp_with_data > 0:
                print(f"--- Checkpoint Progress ({cp_with_data} scenarios with checkpoints) ---")
                print(f"   Avg Checkpoint Progress:        {metrics['avg_checkpoint_progress']}%")
                print(f"")
            else:
                print(f"--- Checkpoint Progress ---")
                print(f"   No checkpoints defined yet")
                print(f"")
        except Exception as e:
            print(f"⚠ print_report error: {e}")
            import traceback
            traceback.print_exc()

        print(f"\n{'Scenario':<40} {'Status':<10} {'Visited':<10} {'Stopped OK':<12} {'Checkpoints':<12} {'Steps':<8} {'Duration':>10}")
        print(f"{'-' * 114}")

        for r in metrics["results"]:
            status = "✅ OK" if r["success"] else "❌ FAIL"
            if r.get("eval_type") in ("url_contains", "url_pattern"):
                visited = "✅" if r.get("visited_eval_url") else "❌"
                stopped = "✅" if r.get("stopped_correctly") else "❌"
            else:
                visited = "—"
                stopped = "—"
            cp_total = r.get("checkpoints_total", 0)
            cp_reached = r.get("checkpoints_reached", 0)
            cp_str = f"{cp_reached}/{cp_total}" if cp_total > 0 else "—"
            duration = r.get("timing", {}).get("duration_sec")
            duration_str = f"{duration:.1f}s" if duration is not None else "—"
            print(
                f"{r['scenario_name']:<40} {status:<10} {visited:<10} {stopped:<12} {cp_str:<12} {r['steps_taken']:<8} {duration_str:>10}")

        print(f"{'#' * 60}\n")

    def save_report(self, metrics: Dict, output_path: str = "benchmark_results.json"):
        """
        Append current benchmark run to benchmark_results.json.

        File format:
        [
          { run_1_metrics... },
          { run_2_metrics... },
          ...
        ]
        """
        existing_runs = []

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                existing_runs = data
            elif isinstance(data, dict):
                # Backward compatibility:
                # if the file contains one old run as a dict, wrap it into a list
                existing_runs = [data]

        except FileNotFoundError:
            existing_runs = []
        except json.JSONDecodeError:
            existing_runs = []

        existing_runs.append(metrics)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(existing_runs, f, ensure_ascii=False, indent=2)

        print(f"Results appended to {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="Run web benchmark with selected agent")

    parser.add_argument(
        "--agent",
        type=str,
        default="openrouter",
        choices=["openrouter", "browser_use"],
        help="Which agent to run",
    )

    parser.add_argument(
        "--benchmarks",
        type=str,
        default="benchmark/benchmarks.json",
        # default="benchmark/test.json",
        help="Path to benchmark scenarios JSON",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )

    parser.add_argument(
        "--cdp-url",
        type=str,
        default="http://localhost:9222",
        help="CDP URL for shared browser agents such as browser-use",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N scenarios (for testing)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (e.g. openai/gpt-4o-mini, anthropic/claude-3.5-sonnet)",
    )

    args = parser.parse_args()

    agent = create_agent(args.agent, model=args.model)

    benchmark = BenchmarkRunner(
        benchmarks_path=args.benchmarks,
        agent=agent,
        headless=args.headless,
        cdp_url=args.cdp_url,
    )

    if args.limit:
        benchmark.scenarios = benchmark.scenarios[:args.limit]


    metrics = await benchmark.run_all()

    benchmark.print_report(metrics)
    benchmark.save_report(metrics)


if __name__ == "__main__":
    asyncio.run(main())