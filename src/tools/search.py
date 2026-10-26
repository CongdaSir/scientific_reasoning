import os

import httpx

from ..config import SearchConfig


class SearchTool:
    def __init__(self, config: SearchConfig, http: httpx.AsyncClient):
        self.config = config
        self.http = http

    async def __call__(self, query: str) -> dict:
        cfg = self.config
        key = os.environ[cfg.api_key_env]
        response = await self.http.post(
            cfg.endpoint,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "query": query,
                "need_content": cfg.need_content,
            },
            timeout=cfg.timeout,
        )
        response.raise_for_status()
        results = []
        for item in response.json()["results"]:
            text = item.get("summary") or ""
            results.append({
                "title": item["title"],
                "url": item["url"],
                "text": text,
            })
        return {"query": query, "results": results}
