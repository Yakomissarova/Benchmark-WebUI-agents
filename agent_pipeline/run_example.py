import asyncio

from agent_pipeline import SitePipeline


async def main():
    # Подставь свою LangChain модель:
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4o-mini")
    llm = None  # <-- замени на реальную модель

    pipeline = SitePipeline(llm=llm, enable_docker=False)

    templates = {
        "index.html": "<!doctype html><html><head><title>X</title></head><body><h1>Hi</h1></body></html>"
    }

    state = await pipeline.process_one("demo-site", templates)
    print("TRACES:", state.traces)
    print("REVIEW:", state.review)
    print("TESTS:", state.tests)


if __name__ == "__main__":
    asyncio.run(main())
