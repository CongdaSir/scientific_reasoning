from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    base_url: str = "http://localhost:8000/v1"
    name: str = "Qwen/Qwen3-8B"
    api_key_env: str = "VLLM_API_KEY"
    timeout: float = 300
    max_tokens: int = 8192
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    enable_thinking: bool = True


@dataclass
class SearchConfig:
    endpoint: str = "https://api.deepseek.com/v1/tools/search"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout: float = 60
    count: int = 5
    need_content: bool = False
    max_result_chars: int = 3000


@dataclass
class CodeConfig:
    timeout: float = 30
    max_output_chars: int = 12000


@dataclass
class ReasoningConfig:
    max_rounds: int = 20


@dataclass
class DatasetConfig:
    question_field: str = "question"
    id_field: str = "id"
    concurrency: int = 4


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) if path else {}
        data = data or {}
        return cls(
            model=ModelConfig(**data.get("model", {})),
            search=SearchConfig(**data.get("search", {})),
            code=CodeConfig(**data.get("code", {})),
            reasoning=ReasoningConfig(**data.get("reasoning", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
        )
