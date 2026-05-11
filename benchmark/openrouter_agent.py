import asyncio
import json
import os
import time
from dotenv import load_dotenv
import httpx
from playwright.async_api import Page
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError, InternalServerError

from benchmark.base_agent import BaseAgent

load_dotenv()
OPEN_ROUTER_MODEL = os.getenv("OPEN_ROUTER_MODEL", "deepseek/deepseek-chat-v3.1")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=httpx.Timeout(120.0, connect=20.0),
    max_retries=5,
)


class OpenRouterAgent(BaseAgent):
    """
    OpenRouter LLM-driven web agent.

    This agent works directly with Playwright page,
    so browser_mode = "playwright_page".
    """

    def __init__(self, name: str | None = None, model: str | None = None):
        self.model = model or OPEN_ROUTER_MODEL
        if name is None:
            name = self.model
        super().__init__(name=name, browser_mode="playwright_page")

    @staticmethod
    def filter_cdp_nodes(nodes):
        """
        Keep only useful interactive elements from CDP accessibility tree.
        This helps reduce noise and save tokens.
        """
        important_elements = []

        for node in nodes:
            role = node.get("role", {}).get("value", "")
            name = node.get("name", {}).get("value", "")
            value = node.get("value", {}).get("value", "")
            node_id = node.get("nodeId")

            if role in [
                "button",
                "link",
                "searchbox",
                "textbox",
                "combobox",
                "checkbox",
                "menuitem",
            ]:
                if name and len(name.strip()) > 1:
                    important_elements.append({
                        "id": node_id,
                        "role": role,
                        "name": name.strip(),
                        "value": value.strip() if value else ""
                    })

        return important_elements

    @staticmethod
    def is_agent_stuck(trajectory, window=5, repeat=4):
        if len(trajectory) < repeat:
            return False

        last = trajectory[-window:]

        # Actual loop: >= repeat identical consecutive actions at the end of the history, considering input text but ignoring 'stop'.
        relevant = [
            s for s in last
            if s.get("action") not in (None, "stop")
        ]
        if len(relevant) >= repeat:
            tail = relevant[-repeat:]
            key = lambda s: (s.get("action"), s.get("element_name"), (s.get("text") or "").strip().lower())
            if all(key(s) == key(tail[0]) for s in tail):
                return True

        # 2) Three errors in a row
        errors_tail = [s.get("error") for s in last[-3:]]
        if len(errors_tail) == 3 and all(e is not None for e in errors_tail):
            return True

        return False

    async def act(
        self,
        page: Page,
        goal: str,
        max_steps: int = 10,
        runtime: dict | None = None,
    ) -> dict:
        """
        Execute the task on a web page.
        """
        self.reset()
        self.log_step(f"Start. Task: {goal}")

        started_at = time.time()
        trajectory = []
        states = [{"step": 0, "url": page.url}]
        last_step = 0

        # Simple token counters for future metrics.
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        try:
            # Create CDP session once and reuse it.
            cdp_session = await page.context.new_cdp_session(page)
            await cdp_session.send("DOM.enable")
            await cdp_session.send("Accessibility.enable")

            for step in range(1, max_steps + 1):
                last_step = step
                self.log_step(f"--- STEP {step}/{max_steps} ---")

                url_before = page.url

                # Read current accessible page structure.
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(1500)
                raw_tree = await cdp_session.send("Accessibility.getFullAXTree")
                nodes = raw_tree.get("nodes", [])
                tree = self.filter_cdp_nodes(nodes)

                prompt = f"""
Ты — AI-автономный веб-агент. Твоя задача: {goal}

Дополнительные инструкции:
- Не надо выполнять вход в систему, переходи напрямую к основной задаче
- ВНИМАТЕЛЬНО СМОТРИ НА ИСТОРИЮ ДЕЙСТВИЙ, если действие было сделано, НЕ ПОВТОРЯЙ
- Если на предыдущем шаге был выполнен ввод, НЕ НАДО его повторять, даже если поле пустое

Вывод (ТОЛЬКО JSON):
- Если видишь конечный результат выполнения задачи - в action пиши "stop"
- НЕ завершай задачу после ввода данных, дождись появления результатов
- Иначе выбери элемент и действие (click или type)
- Формат: {{"element_id": "...", "element_name": "...", "role": "...", "action": "click"|"type"|"stop", "text": "если нужно", "reason": "..."}}

Элементы на текущей странице:
{json.dumps(tree, ensure_ascii=False, indent=2)}

История действий:
{json.dumps(self.history, ensure_ascii=False, indent=2)}
"""

                llm_attempts = 6
                llm_delays = [0, 3, 7, 15, 30, 60]  # секунд перед каждой попыткой
                response = None
                last_llm_error = None

                for attempt_idx, delay in enumerate(llm_delays[:llm_attempts], start=1):
                    if delay:
                        self.log_step(f"⏳ LLM retry sleep {delay}s (attempt {attempt_idx}/{llm_attempts})")
                        await asyncio.sleep(delay)
                    try:
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": "You are a web agent outputting JSON."},
                                {"role": "user", "content": prompt}
                            ],
                            response_format={"type": "json_object"}
                        )
                        if attempt_idx > 1:
                            self.log_step(f"✓ LLM call succeeded on attempt {attempt_idx}/{llm_attempts}")
                        break
                    except (APITimeoutError, APIConnectionError, RateLimitError,
                            InternalServerError, httpx.TimeoutException,
                            httpx.ConnectError, httpx.RemoteProtocolError) as e:
                        last_llm_error = e
                        self.log_step(f"⚠ LLM call failed (attempt {attempt_idx}/{llm_attempts}): {e}")
                        continue

                if response is None:
                    raise last_llm_error if last_llm_error else RuntimeError("LLM call failed without exception")

                # Save token usage if provider returned it.
                if getattr(response, "usage", None):
                    total_prompt_tokens += getattr(response.usage, "prompt_tokens", 0) or 0
                    total_completion_tokens += getattr(response.usage, "completion_tokens", 0) or 0
                    total_tokens += getattr(response.usage, "total_tokens", 0) or 0

                decision = json.loads(response.choices[0].message.content)
                reason = decision.get("reason", "N/A")
                self.log_step(f"[Planning] {reason}")

                action = decision.get("action")
                element_id = decision.get("element_id")
                element_name = decision.get("element_name")
                role = decision.get("role", "link")
                text = decision.get("text", "")
                step_error = None

                # If model says task is complete, stop.
                if action == "stop":
                    self.log_step("Task completed!")

                    url_after = page.url
                    states.append({"step": step, "url": url_after})

                    trajectory.append({
                        "step": step,
                        "element_id": element_id,
                        "element_name": element_name,
                        "role": role,
                        "action": "stop",
                        "text": text,
                        "reason": reason,
                        "url_before": url_before,
                        "url_after": url_after,
                        "changed_state": url_before != url_after,
                        "error": None,
                    })

                    finished_at = time.time()

                    return {
                        "success": True,
                        "steps_taken": step,
                        "final_url": page.url,
                        "error": None,
                        "trajectory": trajectory,
                        "states": states,
                        "timing": {
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "duration_sec": round(finished_at - started_at, 3),
                        },
                        "usage": {
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": total_tokens,
                        },
                        "extra": {
                            "provider": "openrouter",
                            "model": self.model,
                        },
                    }

                # Execute click/type action.
                try:
                    locator = page.get_by_role(role, name=element_name).first

                    if action == "click":
                        await page.bring_to_front()
                        await page.evaluate(
                            "() => { if (document.activeElement && document.activeElement.blur) document.activeElement.blur(); window.focus(); }")
                        await locator.click(timeout=5000)
                        self.log_step(f"[Click] {element_name}")
                        self.history.append(f"Click on: {element_name}")

                        await page.wait_for_timeout(1000)
                        all_pages = page.context.pages
                        if len(all_pages) > 1:
                            new_page = all_pages[-1]
                            if new_page != page:
                                await new_page.bring_to_front()
                                page = new_page
                                cdp_session = await page.context.new_cdp_session(page)
                                await cdp_session.send("DOM.enable")
                                await cdp_session.send("Accessibility.enable")
                                self.log_step(f"[Tab] Switched to new tab: {page.url}")

                    elif action == "type":
                        await locator.fill(text)
                        await page.wait_for_timeout(500)
                        await page.keyboard.press("Enter")
                        await page.wait_for_timeout(500)
                        # await page.keyboard.press("Tab")
                        self.log_step(f"[Type] Text '{text}' in {element_name}")
                        self.history.append(f"Type in {element_name}: {text}")

                    else:
                        # Unknown action from the model.
                        step_error = f"Unknown action: {action}"
                        self.log_step(f"⚠ {step_error}")
                        self.history.append(step_error)

                except Exception as e:
                    step_error = str(e)
                    self.log_step(f"⚠ Error when {action} by {element_name}: {e}")
                    self.history.append(f"Error: {step_error}")

                await page.wait_for_timeout(2000)

                url_after = page.url
                states.append({"step": step, "url": url_after})

                trajectory.append({
                    "step": step,
                    "element_id": element_id,
                    "element_name": element_name,
                    "role": role,
                    "action": action,
                    "text": text,
                    "reason": reason,
                    "url_before": url_before,
                    "url_after": url_after,
                    "changed_state": url_before != url_after,
                    "error": step_error,
                })

                if self.is_agent_stuck(trajectory):
                    self.log_step("⚠ Agent stuck detected — stopping early")

                    finished_at = time.time()

                    return {
                        "success": False,
                        "steps_taken": step,
                        "final_url": page.url,
                        "error": "agent_stuck",
                        "trajectory": trajectory,
                        "states": states,
                        "timing": {
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "duration_sec": round(finished_at - started_at, 3),
                        },
                        "usage": {
                            "prompt_tokens": total_prompt_tokens,
                            "completion_tokens": total_completion_tokens,
                            "total_tokens": total_tokens,
                        },
                        "extra": {
                            "provider": "openrouter",
                            "model": self.model,
                        },
                    }


            # Max steps reached without stop.
            finished_at = time.time()

            return {
                "success": False,
                "steps_taken": max_steps,
                "final_url": page.url,
                "error": "Max steps exceeded without completing the task",
                "trajectory": trajectory,
                "states": states,
                "timing": {
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_sec": round(finished_at - started_at, 3),
                },
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                },
                "extra": {
                    "provider": "openrouter",
                    "model": self.model,
                },
            }

        except Exception as e:
            finished_at = time.time()

            return {
                "success": False,
                "steps_taken": last_step,
                "final_url": page.url if hasattr(page, "url") else "unknown",
                "error": str(e),
                "trajectory": trajectory,
                "states": states,
                "timing": {
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_sec": round(finished_at - started_at, 3),
                },
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_tokens,
                },
                "extra": {
                    "provider": "openrouter",
                    "model": self.model,
                },
            }


# For compatibility / local manual testing
async def run_benchmark_agent(start_url, task_description):
    from playwright.async_api import async_playwright

    agent = OpenRouterAgent()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        await page.goto(start_url, wait_until="commit")
        await page.wait_for_timeout(3000)

        result = await agent.act(page, task_description)
        print("\n=== Result ===")
        print(f"Success: {result['success']}")
        print(f"Steps: {result['steps_taken']}")
        print(f"Final URL: {result['final_url']}")
        print(f"Duration: {result['timing']['duration_sec']} sec")
        print(f"Total tokens: {result['usage']['total_tokens']}")


if __name__ == "__main__":
    # asyncio.run(
    #     run_benchmark_agent(
    #         "https://www.rzd.ru/",
    #         "Начни покупку билета на поезд из Москвы в Санкт-Петербург c 26.06.2026 по 27.06.2026. После заполения необходимых полей нужно нажать Найти. Остановись на моменте оплаты"
    #     )
    # )
    pass