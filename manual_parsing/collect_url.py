import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

HAR_PATH = "../benchmark_data/consultantnews.har"
START_URL = "https://www.consultant.ru/legalnews/"
OUTPUT = "checkpoints_raw.json"

async def collect():
    urls = []

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context()
            page = await context.new_page()
            await page.set_viewport_size({"width": 1280, "height": 800})

            await page.route_from_har(HAR_PATH, update=False)

            def on_nav(frame):
                if frame == page.main_frame:
                    url = frame.url
                    if url and url != "about:blank" and (not urls or urls[-1] != url):
                        urls.append(url)
                        print(f"  [{len(urls)}] {url}")

            page.on("framenavigated", on_nav)
            context.on("page", lambda new_page: new_page.on(
                "framenavigated",
                lambda frame, np=new_page: (
                    urls.append(frame.url) or print(f"  [{len(urls)}] [tab] {frame.url}")
                ) if frame == np.main_frame and frame.url and frame.url != "about:blank" else None
            ))

            await page.goto(START_URL, wait_until="commit")
            print("Проходи сценарий по HAR. Закрой браузер когда закончишь.\n")

            try:
                await page.wait_for_event("close", timeout=0)
            except Exception:
                pass

            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass

    except Exception as e:
        print(f"⚠ Ошибка: {e}")

    finally:
        if not urls:
            print("Нет URL для записи.")
            return

        existing = []
        if Path(OUTPUT).exists():
            try:
                with open(OUTPUT, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing = []

        existing.append(urls)

        with open(OUTPUT, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        print(f"\n{len(urls)} URL добавлено (всего запусков: {len(existing)}) → {OUTPUT}")

if __name__ == "__main__":
    asyncio.run(collect())