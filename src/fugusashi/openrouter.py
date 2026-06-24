from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx


class OpenRouterClient:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://eulogik.com",
                "X-Title": "Fugusashi",
            },
            timeout=120.0,
        )

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        retries: int = 3,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        for attempt in range(retries):
            try:
                resp = self._client.post(
                    f"{self.BASE_URL}/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("choices") and data["choices"][0].get("message", {}).get("content") is None:
                    data["choices"][0]["message"]["content"] = ""
                return data
            except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

    def get_models(self) -> List[Dict[str, Any]]:
        resp = self._client.get(f"{self.BASE_URL}/models")
        resp.raise_for_status()
        return resp.json().get("data", [])

    def get_free_models(self) -> List[Dict[str, Any]]:
        models = self.get_models()
        free = []
        for m in models:
            pricing = m.get("pricing", {})
            if float(pricing.get("prompt", "1") or "1") == 0 and float(pricing.get("completion", "1") or "1") == 0:
                free.append({
                    "id": m["id"],
                    "name": m.get("name", ""),
                    "context_length": m.get("context_length", 0),
                    "max_completion_tokens": m.get("top_provider", {}).get("max_completion_tokens", 0) or 0,
                })
        return sorted(free, key=lambda x: x["context_length"], reverse=True)


COORDINATOR_MODELS = {
    "gpt-oss-20b": "openai/gpt-oss-20b:free",
    "lfm-2.5-1.2b": "liquid/lfm-2.5-1.2b-instruct:free",
    "gpt-oss-120b": "openai/gpt-oss-120b:free",
}

WORKER_MODELS = {
    "gpt-oss-120b": "openai/gpt-oss-120b:free",
    "nemotron-3-ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b:free",
    "hermes-3-405b": "nousresearch/hermes-3-llama-3.1-405b:free",
    "lfm-2.5-1.2b": "liquid/lfm-2.5-1.2b-instruct:free",
}
