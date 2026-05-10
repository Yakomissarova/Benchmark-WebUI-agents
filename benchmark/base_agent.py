from abc import ABC, abstractmethod
from playwright.async_api import Page


class BaseAgent(ABC):
    """
    Base class for all web agents.

    The benchmark interacts with agents via a unified interface.
    Each agent must return a result dict with a stable schema.

    Important:
    The benchmark treats the agent as a black box, but expects
    structured logs (trajectory, states, timing, usage) for metrics.
    """

    def __init__(self, name: str, browser_mode: str = "playwright_page"):
        """
        Args:
            name: Agent name for logs and reports
            browser_mode:
                - "playwright_page" -> agent works directly with Playwright page
                - "shared_cdp" -> agent connects to the same browser via CDP
        """
        self.name = name
        self.browser_mode = browser_mode
        self.step_count = 0
        self.history = []

    @abstractmethod
    async def act(
        self,
        page: Page,
        goal: str,
        max_steps: int = 10,
        runtime: dict | None = None,
    ) -> dict:
        """
        Main agent method used by the benchmark.

        Args:
            page: Playwright page
            goal: Task description
            max_steps: Maximum number of agent steps
            runtime: Optional extra runtime data from benchmark runner

        Returns:
            REQUIRED FIELDS (strict):
            - success: bool
            - steps_taken: int
            - final_url: str
            - error: str | None

            REQUIRED FOR METRICS (must be present, may be empty):
            - trajectory: list[dict]
            - states: list[dict]
            - timing: dict
            - usage: dict

            ------------------------------------------------------------------
            TRAJECTORY SCHEMA (each step MUST contain):

            {
                "step": int,
                "action": str,
                "url_before": str,
                "url_after": str,
                "changed_state": bool,
                "error": str | None,

                # optional but recommended
                "element_id": str | None,
                "element_name": str | None,
                "role": str | None,
                "text": str | None,
                "reason": str | None,
            }

            Notes:
            - changed_state MUST reflect state change (at minimum via URL)
            - trajectory length should correspond to steps_taken

            ------------------------------------------------------------------
            STATES SCHEMA:

            [
                {"step": int, "url": str}
            ]

            Notes:
            - must include initial state (step 0)
            - must be ordered

            ------------------------------------------------------------------
            TIMING SCHEMA:

            {
                "started_at": float | None,
                "finished_at": float | None,
                "duration_sec": float | None
            }

            ------------------------------------------------------------------
            USAGE SCHEMA:

            {
                "prompt_tokens": int | None,
                "completion_tokens": int | None,
                "total_tokens": int | None
            }

            ------------------------------------------------------------------
            EXTRA (optional):
            - any additional debug or provider-specific info

            ------------------------------------------------------------------
            Contract requirements:
            - All required fields MUST be present (even if empty)
            - Benchmark relies on trajectory + states for metrics
        """
        raise NotImplementedError

    def reset(self):
        """Reset agent state before a new scenario."""
        self.step_count = 0
        self.history = []

    def log_step(self, message: str):
        """
        Save a simple text log for debugging.
        This is useful even if the agent later returns structured trajectory.
        """
        self.step_count += 1
        self.history.append(f"[Step {self.step_count}] {message}")
        print(f"  {message}")