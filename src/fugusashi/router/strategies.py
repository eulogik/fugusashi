from __future__ import annotations

import re
from typing import Dict, List

import numpy as np

from .interface import BaseRouter, RouterResult

CODE_KEYWORDS = [
    "python", "javascript", "typescript", "function", "class", "def ",
    "import ", "git ", "docker", "api", "endpoint", "debug", "compile",
    "algorithm", "data structure", "regex", "sql", "query",
]

MATH_KEYWORDS = [
    "calculate", "equation", "derivative", "integral", "theorem",
    "proof", "matrix", "vector", "statistical", "probability",
]

CREATIVE_KEYWORDS = [
    "write a poem", "story", "creative", "brainstorm", "invent",
    "imagine", "metaphor", "narrative",
]


class FallbackRouter(BaseRouter):
    name = "fallback"

    def route(
        self,
        prompt: str,
        messages: List[Dict[str, str]],
        available_models: Dict[str, dict],
        threshold: float = 0.0,
    ) -> RouterResult:
        if not available_models:
            return RouterResult(
                model="",
                confidence=0.0,
                latency_ms=0.0,
                strategy="fallback",
                scores={},
                explanation="No models available",
                needs_escalation=True,
            )

        model_name = next(iter(available_models))
        return RouterResult(
            model=model_name,
            confidence=0.5,
            latency_ms=0.0,
            strategy="fallback",
            scores={m: 0.5 for m in available_models},
            explanation=f"No routing data — defaulting to {model_name}",
        )


class CostRouter(BaseRouter):
    name = "cost"

    def route(
        self,
        prompt: str,
        messages: List[Dict[str, str]],
        available_models: Dict[str, dict],
        threshold: float = 0.0,
    ) -> RouterResult:
        if not available_models:
            return FallbackRouter().route(prompt, messages, available_models, threshold)

        prompt_lower = prompt.lower()
        is_code = any(kw in prompt_lower for kw in CODE_KEYWORDS)
        is_math = any(kw in prompt_lower for kw in MATH_KEYWORDS)
        is_creative = any(kw in prompt_lower for kw in CREATIVE_KEYWORDS)
        need_reasoning = is_code or is_math

        total_cost = sum(
            cfg.get("cost_per_input_token", 0.0) + cfg.get("cost_per_output_token", 0.0)
            for cfg in available_models.values()
        ) or 1.0

        scored = {}
        for name, cfg in available_models.items():
            cost_in = cfg.get("cost_per_input_token", 0.0)
            cost_out = cfg.get("cost_per_output_token", 0.0)
            caps = cfg.get("capabilities", [])
            cost_frac = (cost_in + cost_out) / total_cost
            score = 0.0

            if need_reasoning and "reasoning" in caps:
                score = 0.9 - 0.5 * cost_frac
            elif need_reasoning and "code" in caps:
                score = 0.7 - 0.5 * cost_frac
            elif is_creative and "creative" in caps:
                score = 0.8 - 0.5 * cost_frac
            else:
                score = 0.5 - 0.5 * cost_frac

            score = round(max(0.01, min(1.0, score)), 4)
            scored[name] = score

        best = max(scored, key=scored.get)
        return RouterResult(
            model=best,
            confidence=scored[best],
            latency_ms=0.0,
            strategy="cost",
            scores=scored,
            explanation=f"Routed by capability fit + cost. Top pick: {best}",
            needs_escalation=scored[best] < threshold,
        )


class SimilarityRouter(BaseRouter):
    name = "similarity"

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self._model = None
        self._index = None
        self._index_data: List[dict] = []

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def build_index(self, history: List[dict]):
        if not history:
            return
        texts = [h.get("prompt", "") for h in history]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self._index = np.array(embeddings)
        self._index_data = history

    def route(
        self,
        prompt: str,
        messages: List[Dict[str, str]],
        available_models: Dict[str, dict],
        threshold: float = 0.0,
    ) -> RouterResult:
        if not available_models:
            return FallbackRouter().route(prompt, messages, available_models, threshold)

        if self._index is None or len(self._index_data) == 0:
            return CostRouter().route(prompt, messages, available_models, threshold)

        query_emb = self.model.encode([prompt], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self._index @ query_emb
        top_k = min(5, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        min_similarity = 0.2
        model_votes: Dict[str, list] = {}
        for idx in top_indices:
            entry = self._index_data[idx]
            model_name = entry.get("model", "")
            sim_score = float(scores[idx])
            quality = entry.get("score", 0.5)
            if sim_score >= min_similarity:
                model_votes.setdefault(model_name, []).append(sim_score * quality)

        model_scores = {}
        for name, cfg in available_models.items():
            if name in model_votes and model_votes[name]:
                model_scores[name] = round(float(np.max(model_votes[name])), 4)
            else:
                model_scores[name] = 0.0

        if not model_scores or all(v == 0.0 for v in model_scores.values()):
            return CostRouter().route(prompt, messages, available_models, threshold)

        best = max(model_scores, key=model_scores.get)
        return RouterResult(
            model=best,
            confidence=model_scores[best],
            latency_ms=0.0,
            strategy="similarity",
            scores=model_scores,
            explanation=f"Similarity match from {len(self._index_data)} historical prompts",
            needs_escalation=model_scores[best] < threshold,
        )
