import asyncio
from playwright.async_api import async_playwright

async def run_mirror():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        context = await browser.new_context()
        date = "2026-04-25T12:00:00Z"
        # !!!DATE SCRIPT CAN BRAKE HAR, NEED FIX!!!
        # await context.add_init_script(f"""
        #     const fakeNow = new Date('{date}').getTime();
        #     const NativeDate = window.Date;
        #     window.Date = class extends NativeDate {{
        #         constructor(...args) {{
        #             if (args.length === 0) return new NativeDate(fakeNow);
        #             return new NativeDate(...args);
        #         }}
        #         static now() {{ return fakeNow; }}
        #     }};
        # """)


        page = await context.new_page()

        # Reading from path not from Internet
        await page.route_from_har("benchmark_data/lamoda.har", update=False)

        # Go to the same URL as in recording, but it will be loaded from HAR file, not from Internet
        await page.goto("https://www.lamoda.ru/", wait_until="commit")
        # await page.wait_for_timeout(30000000)
        await page.wait_for_event("close", timeout=0)

        await browser.close()

asyncio.run(run_mirror())