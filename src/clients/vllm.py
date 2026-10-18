import os

import httpx

from ..config import ModelConfig


class VLLMClient:
    def __init__(self, config: ModelConfig, http: httpx.AsyncClient):
        self.config = config
        self.http = http

    async def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        cfg = self.config
        response = await self.http.post(
            f"{cfg.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv(cfg.api_key_env, 'EMPTY')}"},
            json={
                "model": cfg.name,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": cfg.max_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "top_k": cfg.top_k,
                "chat_template_kwargs": {"enable_thinking": cfg.enable_thinking},
                "stream": False,
            },
            timeout=cfg.timeout,
        )
        response.raise_for_status()
        return response.json()
