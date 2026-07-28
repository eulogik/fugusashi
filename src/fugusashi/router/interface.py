from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RouterResult:
    model: str
    confidence: float
    latency_ms: float
    strategy: str
    scores: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    needs_escalation: bool = False


class BaseRouter(ABC):
    name: str = "base"

    @abstractmethod
    def route(
        self,
        prompt: str,
        messages: list[dict[str, str]],
        available_models: dict[str, dict],
        threshold: float = 0.0,
    ) -> RouterResult:
        ...

    def _measure(self, fn, *args, **kwargs):
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start) * 1000
        return result, elapsed
