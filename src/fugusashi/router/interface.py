from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List
import time


@dataclass
class RouterResult:
    model: str
    confidence: float
    latency_ms: float
    strategy: str
    scores: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    needs_escalation: bool = False


class BaseRouter(ABC):
    name: str = "base"

    @abstractmethod
    def route(
        self,
        prompt: str,
        messages: List[Dict[str, str]],
        available_models: Dict[str, dict],
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
