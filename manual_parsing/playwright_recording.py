import asyncio
import json
import os
from pathlib import Path
from playwright.async_api import async_playwright

# Configuration (write your recording details here)
RECORDING_CONFIG = {
    "id": "test",
    "name": "test",
    "har_path": "benchmark_data/lamoda.har",
    "start_url": "https://www.lamoda.ru/",
    "goal": "test",
    "eval_type": "url_contains",  # Possible values: "url_contains"
}

RECORDINGS_JSON_PATH = "benchmark_data/recordings.json"
DATE = "2026-04-25T12:00:00Z"

async def save_recording_metadata(final_url: str):
    """Saves the recording metadata to a JSON file."""
    recording_entry = {
        **RECORDING_CONFIG,
        "eval_value": final_url
    }

    Path(RECORDINGS_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)

    if os.path.exists(RECORDINGS_JSON_PATH):
        with open(RECORDINGS_JSON_PATH, 'r', encoding='utf-8') as f:
            recordings = json.load(f)
    else:
        recordings = []

    # Rewrite existing entry if it exists, otherwise append
    existing_index = next((i for i, r in enumerate(recordings) if r.get("id") == RECORDING_CONFIG["id"]), None)
    if existing_index is not None:
        recordings[existing_index] = recording_entry
    else:
        recordings.append(recording_entry)

    with open(RECORDINGS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(recordings, f, ensure_ascii=False, indent=2)

async def record():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        context = await browser.new_context(record_har_path=RECORDING_CONFIG["har_path"])
        page = await context.new_page()

        await page.goto(RECORDING_CONFIG["start_url"])

        await page.wait_for_event("close", timeout=0)

        final_url = page.url

        await context.close()
        await browser.close()

        await save_recording_metadata(final_url)

if __name__ == "__main__":
    asyncio.run(record())
