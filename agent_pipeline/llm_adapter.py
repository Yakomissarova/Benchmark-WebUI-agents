from __future__ import annotations

import json
import re
from typing import Any, Type

from pydantic import BaseModel


class LLMAdapter:
    """
    Wrap a LangChain chat model and enforce JSON-only structured outputs via Pydantic schemas.

    Requirement: llm provides `await llm.ainvoke(messages)` returning object with `.content`
    (or a string-like response).
    """

    def __init__(self, llm: Any):
        self.llm = llm

    async def ajson(self, system: str, user: str, schema: Type[BaseModel]) -> BaseModel:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = await self.llm.ainvoke(messages)
        content = resp.content if hasattr(resp, "content") else str(resp)

        # Extract first JSON block (object or array)
        m = re.search(r"(\{.*\}|\[.*\])", content, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Model did not return JSON. Got:\n{content}")

        data = json.loads(m.group(1))
        return schema.model_validate(data)
