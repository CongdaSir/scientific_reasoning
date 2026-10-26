from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    base_url: str | None = None
    name: str | None = None
    api_key_env: str | None = None
    timeout: float = 300
    max_tokens: int | None = None
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    enable_thinking: bool = True
    provider: str = "vllm"
    reasoning_effort: str | None = None

    def __post_init__(self):
        defaults = {
            "vllm": ("http://localhost:8000/v1", "Qwen/Qwen3-8B", "VLLM_API_KEY", 8192),
            "deepseek": ("https://api.deepseek.com/v1", "deepseek-flash", "DEEPSEEK_API_KEY", 393216),
        }
        if self.provider not in defaults:
            raise ValueError(f"Unknown model.provider: {self.provider!r}; expected 'vllm' or 'deepseek'")
        for field_name, default in zip(("base_url", "name", "api_key_env", "max_tokens"), defaults[self.provider]):
            if getattr(self, field_name) is None:
                setattr(self, field_name, default)


@dataclass
class SearchConfig:
    endpoint: str = "https://api.deepseek.com/v1/tools/search"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout: float = 60
    need_content: bool = False
    # Accepted for older config files; summaries are always returned in full.
    max_result_chars: int | None = None


@dataclass
class CodeConfig:
    timeout: float = 30
    auto_install: bool = True
    install_timeout: float = 300
    package_map: dict[str, str] = field(default_factory=dict)
    # Accepted for older config files; both output streams are returned in full.
    max_output_chars: int | None = None


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
