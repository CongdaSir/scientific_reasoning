import os

import httpx

from ..config import SearchConfig


class SearchTool:
    def __init__(self, config: SearchConfig, http: httpx.AsyncClient):
        self.config = config
        self.http = http

    async def __call__(self, query: str, count: int | None = None) -> dict:
        cfg = self.config
        key = os.environ[cfg.api_key_env]
        response = await self.http.post(
            cfg.endpoint,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "query": query,
                "count": cfg.count if count is None else count,
                "need_content": cfg.need_content,
            },
            timeout=cfg.timeout,
        )
        response.raise_for_status()
        results = []
        for item in response.json()["results"]:
            text = item.get("content") or item.get("summary") or item.get("snippet", "")
            results.append({
                "title": item["title"],
                "url": item["url"],
                "text": text[:cfg.max_result_chars],
                "truncated": len(text) > cfg.max_result_chars,
            })
        return {"query": query, "results": results}
