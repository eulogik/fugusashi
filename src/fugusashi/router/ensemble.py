from __future__ import annotations

import time
from typing import Dict, List

from .interface import BaseRouter, RouterResult
from .strategies import CostRouter, FallbackRouter, SimilarityRouter


class EnsembleRouter(BaseRouter):
    name = "ensemble"

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        confidence_threshold: float = 0.4,
        fallback_model: str = "default",
        prefer_local: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.fallback_model = fallback_model
        self.prefer_local = prefer_local
        self.similarity_router = SimilarityRouter(embedding_model=embedding_model)
        self.cost_router = CostRouter()
        self.fallback_router = FallbackRouter()

    def build_index(self, history: List[dict]):
        self.similarity_router.build_index(history)

    def route(
        self,
        prompt: str,
        messages: List[Dict[str, str]],
        available_models: Dict[str, dict],
        threshold: float = 0.0,
    ) -> RouterResult:
        effective_threshold = threshold or self.confidence_threshold
        strategies = [
            ("similarity", self.similarity_router),
            ("cost", self.cost_router),
        ]

        best_result = None

        for name, router in strategies:
            start = time.perf_counter()
            result = router.route(prompt, messages, available_models, effective_threshold)
            elapsed = (time.perf_counter() - start) * 1000
            result.latency_ms = round(elapsed, 2)

            if (
                best_result is None
                and not result.needs_escalation
                and result.confidence >= effective_threshold
            ):
                best_result = result

            if best_result is not None:
                break

        if best_result is None:
            all = []
            for name, router in strategies:
                result = router.route(prompt, messages, available_models, 0.0)
                all.append(result)
            best_result = max(all, key=lambda r: r.confidence) if all else None

        if best_result is None:
            best_result = self.fallback_router.route(prompt, messages, available_models, effective_threshold)

        best_result.strategy = f"ensemble({best_result.strategy})"

        if best_result.confidence < effective_threshold:
            best_result.explanation += (
                f" | Confidence {best_result.confidence:.2f} below threshold "
                f"{effective_threshold:.2f} — escalation recommended"
            )

        return best_result
