from __future__ import annotations

from typing import Any
from dotenv import load_dotenv
import os
from playwright.async_api import Page

from benchmark.base_agent import BaseAgent

# Browser Use imports
from browser_use import Agent, Browser, ChatBrowserUse, ChatOpenAI

load_dotenv()


class BrowserUseAgent(BaseAgent):
    """
    Adapter for browser-use.

    Externally it looks like any other benchmark agent:
        await act(page, goal, max_steps, runtime)

    Internally it connects browser-use to the same Chrome via CDP.
    """

    def __init__(self, name: str = "BrowserUse Agent", model: str = os.getenv("OPEN_ROUTER_MODEL")):
        super().__init__(name=name, browser_mode="shared_cdp")
        self.model = model or "bu-latest"

    def _to_python(self, obj: Any) -> Any:
        """
        Recursively convert browser-use / pydantic / custom objects
        to plain JSON-serializable Python data.
        """
        if obj is None:
            return None

        if isinstance(obj, (str, int, float, bool)):
            return obj

        if isinstance(obj, list):
            return [self._to_python(x) for x in obj]

        if isinstance(obj, tuple):
            return [self._to_python(x) for x in obj]

        if isinstance(obj, dict):
            return {str(k): self._to_python(v) for k, v in obj.items()}

        if hasattr(obj, "model_dump"):
            try:
                return self._to_python(obj.model_dump(mode="json"))
            except Exception:
                pass

        if hasattr(obj, "dict"):
            try:
                return self._to_python(obj.dict())
            except Exception:
                pass

        if hasattr(obj, "__dict__"):
            try:
                return self._to_python(vars(obj))
            except Exception:
                pass

        # Last safe fallback
        return str(obj)

    def _normalize_action_name(self, raw_name: str | None) -> str:
        """
        Convert browser-use action names to a small common set
        that can be used by the benchmark across different agents.
        """
        if not raw_name:
            return "other"

        raw_name = raw_name.lower()

        if raw_name == "click":
            return "click"
        if raw_name in {"input", "type"}:
            return "type"
        if raw_name in {"done", "stop"}:
            return "stop"
        if raw_name in {"navigate", "go_to_url", "open_tab", "search_google"}:
            return "navigate"
        if raw_name == "wait":
            return "wait"
        if raw_name in {"extract_content", "extract"}:
            return "extract"

        return "other"

    def _extract_reason(self, thought_obj: Any) -> str:
        """
        Try to build a readable step reason from browser-use thoughts.
        Exact fields may vary by version.
        """
        thought = self._to_python(thought_obj)

        if isinstance(thought, str):
            return thought

        if not isinstance(thought, dict):
            return ""

        parts = []
        for key in ["next_goal", "memory", "evaluation_previous_goal", "thought", "summary"]:
            value = thought.get(key)
            if value:
                parts.append(f"{key}: {value}")

        return " | ".join(parts)

    def _extract_action_payload(self, raw_action: Any) -> tuple[str, dict, Any]:
        """
        Extract:
        - action name
        - action parameters
        - raw action payload

        browser-use action objects can differ a bit across versions,
        so this parser is intentionally tolerant.
        """
        payload = self._to_python(raw_action)

        if not isinstance(payload, dict):
            return "other", {}, payload

        known_keys = [
            "click",
            "input",
            "type",
            "done",
            "navigate",
            "go_to_url",
            "open_tab",
            "wait",
            "scroll",
            "extract_content",
            "extract",
            "search_google",
            "send_keys",
            "select_dropdown_option",
        ]

        # Most common shape: {"click": {...}} or {"input": {...}}
        for key in known_keys:
            if key in payload:
                params = self._to_python(payload[key])
                if not isinstance(params, dict):
                    params = {"value": params}
                return key, params, payload

        # Fallback shape: {"action_name": "...", "params": {...}}
        if "action_name" in payload:
            raw_name = payload.get("action_name")
            params = payload.get("params") or payload.get("arguments") or {}
            params = self._to_python(params)
            if not isinstance(params, dict):
                params = {"value": params}
            return str(raw_name), params, payload

        # Last fallback: single-key dict
        if len(payload) == 1:
            raw_name = next(iter(payload.keys()))
            params = self._to_python(payload[raw_name])
            if not isinstance(params, dict):
                params = {"value": params}
            return raw_name, params, payload

        return "other", payload, payload

    async def act(
        self,
        page: Page,
        goal: str,
        max_steps: int = 10,
        runtime: dict | None = None,
    ) -> dict:
        """
        Run browser-use on the same Chrome instance via CDP.

        Expected runtime:
            {
                "cdp_url": "http://localhost:9222",
                ...
            }
        """
        self.reset()

        runtime = runtime or {}
        cdp_url = runtime.get("cdp_url")

        if not cdp_url:
            return {
                "success": False,
                "steps_taken": 0,
                "final_url": page.url,
                "error": "BrowserUseAgent requires runtime['cdp_url']",
                "trajectory": [],
                "states": [{"step": 0, "url": page.url}],
                "timing": {},
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "cost": None,
                },
                "extra": {},
            }

        initial_url = page.url

        try:
            # Connect browser-use to the same Chrome via CDP.
            browser = Browser(cdp_url=cdp_url)

            if self.model in ["bu-1-0", "bu-latest"]:
                llm = ChatBrowserUse(model=self.model)
            else:
                llm = ChatOpenAI(
                    model=self.model,
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "http://localhost:3000", # necessary for OpenRouter
                        "X-Title": "Web-UI-Agent-Benchmark"
                    }
                )

            agent = Agent(
                task=goal,
                llm=llm,
                browser=browser,
            )

            history = await agent.run(max_steps=max_steps)

            urls = history.urls() if hasattr(history, "urls") else []
            action_names = history.action_names() if hasattr(history, "action_names") else []
            model_actions = history.model_actions() if hasattr(history, "model_actions") else []
            thoughts = history.model_thoughts() if hasattr(history, "model_thoughts") else []
            errors = history.errors() if hasattr(history, "errors") else []

            # Build states from visited URLs
            states = [{"step": 0, "url": initial_url}]
            for i, url in enumerate(urls, start=1):
                states.append({"step": i, "url": url})

            # Build normalized trajectory
            trajectory = []
            previous_url = initial_url

            # We treat one browser-use "step" as one trajectory event for now.
            # This is the simplest stable version.
            max_len = max(len(action_names), len(model_actions), len(errors), len(urls), len(thoughts))

            for i in range(max_len):
                raw_action = model_actions[i] if i < len(model_actions) else None
                raw_name = action_names[i] if i < len(action_names) else None
                thought = thoughts[i] if i < len(thoughts) else None
                step_error = errors[i] if i < len(errors) else None
                url_after = urls[i] if i < len(urls) else previous_url

                parsed_name, params, raw_payload = self._extract_action_payload(raw_action)

                # Prefer explicit action_names() when available
                effective_name = raw_name or parsed_name
                action = self._normalize_action_name(effective_name)

                text = (
                    params.get("text")
                    or params.get("value")
                    or params.get("input_text")
                    or params.get("query")
                    or ""
                )

                element_id = params.get("index") or params.get("element_id") or params.get("id")
                if element_id is not None:
                    element_id = str(element_id)

                trajectory.append({
                    "step": i + 1,
                    "element_id": element_id,
                    "element_name": None,   # browser-use does not always expose this directly
                    "role": None,           # same here
                    "action": action,
                    "text": text,
                    "reason": self._extract_reason(thought),
                    "url_before": previous_url,
                    "url_after": url_after,
                    "changed_state": previous_url != url_after,
                    "error": str(step_error) if step_error else None,
                    "raw_action": raw_payload,
                })

                previous_url = url_after

            final_url = urls[-1] if urls else page.url

            if trajectory:
                trajectory[-1]["url_after"] = final_url

            states.append({"step": len(trajectory), "url": final_url})

            # browser-use exposes success helpers on history
            history_success = None
            if hasattr(history, "is_successful"):
                try:
                    history_success = history.is_successful()
                except Exception:
                    history_success = None

            steps_taken = 0
            if hasattr(history, "number_of_steps"):
                try:
                    steps_taken = history.number_of_steps()
                except Exception:
                    steps_taken = len(trajectory)
            else:
                steps_taken = len(trajectory)

            duration_sec = None
            if hasattr(history, "total_duration_seconds"):
                try:
                    duration_sec = history.total_duration_seconds()
                except Exception:
                    duration_sec = None

            error_message = None
            non_empty_errors = [str(e) for e in errors if e]
            if non_empty_errors:
                error_message = non_empty_errors[-1]

            return {
                "success": bool(history_success),
                "steps_taken": steps_taken,
                "final_url": final_url,
                "error": error_message,
                "trajectory": trajectory,
                "states": states,
                "timing": {
                    "started_at": None,
                    "finished_at": None,
                    "duration_sec": round(duration_sec, 3) if duration_sec is not None else None,
                },
                "usage": {
                    # We keep token fields for compatibility with future metrics.
                    # In the minimal version we do not collect them yet.
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "cost": None,
                },
                "extra": {
                    "provider": "browser_use",
                    "model": self.model,
                    "history_done": history.is_done() if hasattr(history, "is_done") else None,
                },
            }

        except Exception as e:
            return {
                "success": False,
                "steps_taken": 0,
                "final_url": page.url,
                "error": str(e),
                "trajectory": [],
                "states": [{"step": 0, "url": page.url}],
                "timing": {},
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                    "cost": None,
                },
                "extra": {
                    "provider": "browser_use",
                    "model": self.model,
                },
            }