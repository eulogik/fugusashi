from __future__ import annotations

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, YamlConfigSettingsSource


class ModelConfig(BaseSettings):
    name: str
    provider: str = "openai"
    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    max_tokens: int = 8192
    capabilities: List[str] = Field(default_factory=lambda: ["chat"])
    weight: float = 1.0
    rpm: int = 1000
    tpm: int = 100000
    description: str = ""


class RouterStrategy(BaseSettings):
    name: str = "ensemble"
    confidence_threshold: float = 0.4
    fallback_model: str = "default"
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_top_k: int = 5
    prefer_local: bool = True


class Tier1Config(BaseSettings):
    enabled: bool = True
    overhead_ms_max: int = 20
    router: RouterStrategy = RouterStrategy()


class Tier2Config(BaseSettings):
    enabled: bool = False


class ObservabilityConfig(BaseSettings):
    log_routing_decisions: bool = True
    log_model_calls: bool = True
    expose_transparency_endpoint: bool = True
    log_level: str = "INFO"


class AppConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 6060
    api_key: Optional[str] = None
    default_model: str = "gpt-4o-mini"
    models: List[ModelConfig] = Field(default_factory=list)
    tier1: Tier1Config = Tier1Config()
    tier2: Tier2Config = Tier2Config()
    observability: ObservabilityConfig = ObservabilityConfig()

    @classmethod
    def from_yaml(cls, path: str) -> AppConfig:
        yaml_settings = YamlConfigSettingsSource(
            cls, yaml_file=path, yaml_file_encoding="utf-8"
        )
        return cls(**yaml_settings())

    model_config = {"arbitrary_types_allowed": True}
