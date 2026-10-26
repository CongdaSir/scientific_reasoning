import httpx

from ..config import ModelConfig
from .deepseek import DeepSeekClient
from .vllm import VLLMClient


def create_model_client(config: ModelConfig, http: httpx.AsyncClient) -> VLLMClient | DeepSeekClient:
    if config.provider == "vllm":
        return VLLMClient(config, http)
    if config.provider == "deepseek":
        return DeepSeekClient(config, http)
    raise ValueError(f"Unknown model.provider: {config.provider!r}")
