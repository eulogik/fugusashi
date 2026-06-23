from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import litellm
from litellm import Router as LiteLLMRouter
from litellm.utils import get_llm_provider


def build_litellm_router(model_configs: List[Dict[str, Any]]) -> LiteLLMRouter:
    model_list = []
    for cfg in model_configs:
        provider = cfg.get("provider", "openai")
        model_name = cfg.get("model", "")
        entry = {
            "model_name": cfg.get("name", model_name),
            "litellm_params": {
                "model": f"{provider}/{model_name}" if provider != "openai" else model_name,
                "max_tokens": cfg.get("max_tokens", 8192),
                "rpm": cfg.get("rpm", 1000),
                "tpm": cfg.get("tpm", 100000),
            },
        }
        if cfg.get("api_base"):
            entry["litellm_params"]["api_base"] = cfg["api_base"]
        if cfg.get("api_key"):
            entry["litellm_params"]["api_key"] = cfg["api_key"]
        model_list.append(entry)

    return LiteLLMRouter(model_list=model_list)


class ModelClient:
    def __init__(self, model_configs: List[Dict[str, Any]]):
        self.model_configs = {c.get("name", c["model"]): c for c in model_configs}
        self._router = build_litellm_router(model_configs)

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        return self.model_configs

    async def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Tuple[Any, float, int, int, str]:
        config = self.model_configs.get(model_name, {})
        provider_model = config.get("model", model_name)
        provider = config.get("provider", "openai")

        litellm_model = (
            f"{provider}/{provider_model}"
            if provider not in ("openai", "") and "/" not in provider_model
            else provider_model
        )

        kwargs = {
            "model": litellm_model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if config.get("api_base"):
            kwargs["api_base"] = config["api_base"]

        start = time.perf_counter()
        try:
            response = await self._router.acompletion(**kwargs)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            raise RuntimeError(f"Model call failed: {e}") from e
        finally:
            elapsed = (time.perf_counter() - start) * 1000

        if stream:
            return response, elapsed, 0, 0, provider

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        cost = litellm.completion_cost(response)
        return response, elapsed, prompt_tokens, completion_tokens, provider

    async def call_model_stream(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[bytes, None]:
        config = self.model_configs.get(model_name, {})
        provider_model = config.get("model", model_name)
        provider = config.get("provider", "openai")

        litellm_model = (
            f"{provider}/{provider_model}"
            if provider not in ("openai", "") and "/" not in provider_model
            else provider_model
        )

        kwargs = {
            "model": litellm_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if config.get("api_base"):
            kwargs["api_base"] = config["api_base"]

        async for chunk in await self._router.acompletion(**kwargs):
            yield chunk
