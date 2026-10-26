import os

import httpx

from ..config import ModelConfig


class DeepSeekClient:
    """DeepSeek Chat Completions with the same tool protocol as the vLLM client."""

    def __init__(self, config: ModelConfig, http: httpx.AsyncClient):
        self.config = config
        self.http = http

    async def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        cfg = self.config
        key = os.getenv(cfg.api_key_env, "").strip()
        if not key:
            raise ValueError(f"DeepSeek API key is missing; set environment variable {cfg.api_key_env}")
        payload = {
            "model": cfg.name,
            "messages": messages,
            "max_tokens": cfg.max_tokens,
            "thinking": {"type": "enabled" if cfg.enable_thinking else "disabled"},
            "top_p": cfg.top_p,
            "stream": False,
        }
        if tools:
            payload.update({"tools": tools, "tool_choice": "auto"})
        if cfg.enable_thinking:
            if cfg.reasoning_effort is not None:
                payload["reasoning_effort"] = cfg.reasoning_effort
        else:
            payload["temperature"] = cfg.temperature
        response = await self.http.post(
            f"{cfg.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=payload,
            timeout=cfg.timeout,
        )
        response.raise_for_status()
        return response.json()
