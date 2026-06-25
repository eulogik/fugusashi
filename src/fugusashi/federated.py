from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .coordinator import CMAESRouter, RouteDecision
from .openrouter import WORKER_MODELS


@dataclass
class LocalUpdate:
    client_id: str
    gradient: np.ndarray
    n_samples: int
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FederatedRound:
    round_id: int
    global_weights: np.ndarray
    participating_clients: List[str]
    aggregated_gradient: np.ndarray
    timestamp: float


class FederatedRouter:
    def __init__(
        self,
        n_models: int = 5,
        embed_dim: int = 384,
        noise_multiplier: float = 0.1,
        min_clients: int = 3,
        data_dir: str = ".fugusashi_data",
    ):
        self.n_models = n_models
        self.embed_dim = embed_dim
        self.n_params = embed_dim + 1
        self.noise_multiplier = noise_multiplier
        self.min_clients = min_clients
        self.data_dir = data_dir

        self.global_weights = np.zeros(self.n_params)
        self.current_round = 0
        self._pending_updates: List[LocalUpdate] = []
        self._round_history: List[FederatedRound] = []
        self._client_registry: Dict[str, Dict[str, Any]] = {}

    def register_client(self, client_id: str, metadata: Optional[Dict[str, Any]] = None):
        self._client_registry[client_id] = {
            "registered_at": time.time(),
            "n_updates": 0,
            "last_update": None,
            "metadata": metadata or {},
        }

    def create_local_router(self, client_id: str) -> CMAESRouter:
        router = CMAESRouter(
            model_names=list(WORKER_MODELS.values())[:self.n_models],
            embed_dim=self.embed_dim,
        )
        router.mean = self.global_weights.copy()
        router.best_params = self.global_weights.copy()
        return router

    def submit_update(
        self,
        client_id: str,
        local_weights: np.ndarray,
        n_samples: int,
        metrics: Optional[Dict[str, float]] = None,
    ):
        if client_id not in self._client_registry:
            self.register_client(client_id)

        gradient = local_weights - self.global_weights

        noisy_gradient = gradient + np.random.normal(
            0, self.noise_multiplier * np.std(gradient) + 1e-8, size=gradient.shape
        )

        update = LocalUpdate(
            client_id=client_id,
            gradient=noisy_gradient,
            n_samples=n_samples,
            timestamp=time.time(),
            metrics=metrics or {},
        )
        self._pending_updates.append(update)
        self._client_registry[client_id]["n_updates"] += 1
        self._client_registry[client_id]["last_update"] = time.time()

        return {"status": "accepted", "pending_updates": len(self._pending_updates)}

    def should_aggregate(self) -> bool:
        unique_clients = len(set(u.client_id for u in self._pending_updates))
        return unique_clients >= self.min_clients

    def aggregate(self) -> Optional[FederatedRound]:
        if not self.should_aggregate():
            return None

        total_samples = sum(u.n_samples for u in self._pending_updates)
        weighted_gradient = np.zeros(self.n_params)

        for update in self._pending_updates:
            weight = update.n_samples / total_samples
            weighted_gradient += weight * update.gradient

        self.global_weights += weighted_gradient
        self.current_round += 1

        round_info = FederatedRound(
            round_id=self.current_round,
            global_weights=self.global_weights.copy(),
            participating_clients=list(set(u.client_id for u in self._pending_updates)),
            aggregated_gradient=weighted_gradient,
            timestamp=time.time(),
        )
        self._round_history.append(round_info)
        self._pending_updates = []

        return round_info

    def get_client_stats(self) -> Dict[str, Any]:
        return {
            "n_clients": len(self._client_registry),
            "current_round": self.current_round,
            "pending_updates": len(self._pending_updates),
            "clients": {
                cid: {
                    "n_updates": info["n_updates"],
                    "last_update": info["last_update"],
                }
                for cid, info in self._client_registry.items()
            },
        }

    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.data_dir, "federated_state.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "global_weights": self.global_weights.tolist(),
            "current_round": self.current_round,
            "n_clients": len(self._client_registry),
            "round_history": [
                {
                    "round_id": r.round_id,
                    "participating_clients": r.participating_clients,
                    "timestamp": r.timestamp,
                }
                for r in self._round_history
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.data_dir, "federated_state.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.global_weights = np.array(data["global_weights"])
        self.current_round = data.get("current_round", 0)


class RoutingExplainer:
    def __init__(self, model_names: Optional[List[str]] = None):
        if model_names is None:
            model_names = [n.split("/")[-1].split(":")[0] for n in WORKER_MODELS.values()]
        self.model_names = model_names
        self._capability_map = self._build_capability_map()

    def _build_capability_map(self) -> Dict[str, List[str]]:
        return {
            "gpt-oss-120b": ["complex reasoning", "code generation", "long context"],
            "nemotron-3-ultra-550b-a55b": ["code generation", "math", "reasoning"],
            "nemotron-3-super-120b-a12b": ["code generation", "reasoning", "instruction following"],
            "hermes-3-llama-3.1-405b": ["creative writing", "reasoning", "general knowledge"],
            "lfm-2.5-1.2b-instruct": ["fast responses", "simple tasks", "low latency"],
        }

    def explain(self, prompt: str, decision: RouteDecision, scores: Dict[str, float]) -> str:
        chosen_short = decision.model.split("/")[-1].split(":")[0]
        capabilities = self._capability_map.get(chosen_short, ["general tasks"])

        prompt_features = self._analyze_prompt(prompt)

        lines = [
            f"**Decision:** Route to `{chosen_short}` (confidence: {decision.confidence:.1%})",
            "",
            f"**Why:** This prompt involves {prompt_features}. "
            f"`{chosen_short}` is best suited for {capabilities[0]}.",
            "",
            "**Alternatives considered:**",
        ]

        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        for model, score in sorted_scores[1:4]:
            if score > 0.1:
                alt_caps = self._capability_map.get(model, ["general"])
                lines.append(f"- `{model}` ({score:.1%}): better for {alt_caps[0]}")

        lines.append("")
        lines.append(f"**Latency:** {decision.latency_ms:.1f}ms | **Strategy:** {decision.strategy}")

        return "\n".join(lines)

    def _analyze_prompt(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        features = []

        code_keywords = ["function", "class", "code", "python", "javascript", "debug", "script", "implement", "algorithm"]
        if any(kw in prompt_lower for kw in code_keywords):
            features.append("code generation")

        math_keywords = ["calculate", "equation", "math", "theorem", "proof", "derivative"]
        if any(kw in prompt_lower for kw in math_keywords):
            features.append("mathematical reasoning")

        creative_keywords = ["poem", "story", "creative", "write", "imagine", "invent"]
        if any(kw in prompt_lower for kw in creative_keywords):
            features.append("creative writing")

        factual_keywords = ["what is", "who is", "capital", "meaning", "explain", "define"]
        if any(kw in prompt_lower for kw in factual_keywords):
            features.append("factual knowledge")

        if not features:
            features.append("general language understanding")

        return " and ".join(features)
