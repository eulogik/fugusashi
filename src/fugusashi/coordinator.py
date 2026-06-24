from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .openrouter import OpenRouterClient, WORKER_MODELS


@dataclass
class Task:
    prompt: str
    category: str = "general"
    embedding: Optional[np.ndarray] = None


@dataclass
class RouteDecision:
    model: str
    confidence: float
    scores: Dict[str, float]
    strategy: str
    latency_ms: float


@dataclass
class Plan:
    subtasks: List[Dict[str, Any]]
    fitness: float = 0.0


class PromptEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]


class CMAESRouter:
    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        embed_dim: int = 384,
        population_size: int = 20,
        n_generations: int = 50,
        sigma_init: float = 0.3,
        data_dir: str = ".fugusashi_data",
    ):
        if model_names is None:
            model_names = list(WORKER_MODELS.values())
        self.model_names = model_names
        self.n_models = len(model_names)
        self.embed_dim = embed_dim
        self.population_size = population_size
        self.n_generations = n_generations
        self.sigma_init = sigma_init
        self.data_dir = data_dir

        self.embedder = PromptEmbedder()
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.client = OpenRouterClient(api_key=self.api_key) if self.api_key else None

        self.n_params = embed_dim + 1
        self.mean = np.zeros(self.n_params)
        self.sigma = sigma_init
        self.best_fitness = -float("inf")
        self.best_params: Optional[np.ndarray] = None
        self._generation = 0
        self._history: List[Dict[str, Any]] = []

    def _predict(self, params: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        weights = params[:self.embed_dim]
        bias = params[self.embed_dim]
        logits = embedding * weights + bias
        if len(logits) > self.n_models:
            logits = logits[:self.n_models]
        elif len(logits) < self.n_models:
            logits = np.concatenate([logits, np.zeros(self.n_models - len(logits))])
        exp = np.exp(logits - np.max(logits))
        return exp / exp.sum()

    def _sample_population(self) -> List[np.ndarray]:
        return [
            self.mean + self.sigma * np.random.randn(self.n_params)
            for _ in range(self.population_size)
        ]

    def _evaluate_fast(
        self,
        params: np.ndarray,
        task: Task,
    ) -> float:
        if task.embedding is None:
            task.embedding = self.embedder.embed(task.prompt)
        probs = self._predict(params, task.embedding)
        chosen_idx = int(np.argmax(probs))
        return float(probs[chosen_idx])

    def _evaluate(
        self,
        params: np.ndarray,
        task: Task,
    ) -> float:
        if task.embedding is None:
            task.embedding = self.embedder.embed(task.prompt)

        probs = self._predict(params, task.embedding)
        chosen_idx = int(np.argmax(probs))
        chosen_model = self.model_names[chosen_idx]

        if self.client is None:
            return float(probs[chosen_idx])

        try:
            result = self.client.chat_completion(
                model=chosen_model,
                messages=[{"role": "user", "content": task.prompt}],
                max_tokens=100,
            )
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            quality = min(len(content) / 200, 1.0) if content else 0.0
            efficiency = 1.0 - min(completion_tokens / 200, 1.0)
            confidence = float(probs[chosen_idx])

            return quality * 0.5 + efficiency * 0.2 + confidence * 0.3
        except Exception:
            return 0.0

    def evolve(self, tasks: List[Task], fast: bool = True) -> None:
        if not tasks:
            return

        eval_fn = self._evaluate_fast if fast else self._evaluate

        for gen in range(self.n_generations):
            population = self._sample_population()
            fitnesses = []

            for params in population:
                task_fitnesses = []
                for task in tasks[:5]:
                    f = eval_fn(params, task)
                    task_fitnesses.append(f)
                fitnesses.append(np.mean(task_fitnesses))

            fitnesses = np.array(fitnesses)
            sorted_idx = np.argsort(fitnesses)[::-1]
            mu = self.population_size // 2

            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / weights.sum()

            new_mean = np.zeros(self.n_params)
            for i in range(mu):
                new_mean += weights[i] * population[sorted_idx[i]]

            self.mean = new_mean
            self.sigma *= 0.97

            best_idx = sorted_idx[0]
            if fitnesses[best_idx] > self.best_fitness:
                self.best_fitness = float(fitnesses[best_idx])
                self.best_params = population[best_idx].copy()

            self._generation += 1
            self._history.append({
                "generation": self._generation,
                "best_fitness": float(fitnesses[best_idx]),
                "mean_fitness": float(np.mean(fitnesses)),
                "sigma": float(self.sigma),
            })

    def route(self, prompt: str) -> RouteDecision:
        start = time.perf_counter()
        embedding = self.embedder.embed(prompt)

        params = self.best_params if self.best_params is not None else self.mean
        probs = self._predict(params, embedding)

        chosen_idx = int(np.argmax(probs))
        chosen_model = self.model_names[chosen_idx]

        scores = {}
        for i, name in enumerate(self.model_names):
            short = name.split("/")[-1].split(":")[0]
            scores[short] = round(float(probs[i]), 4)

        elapsed = (time.perf_counter() - start) * 1000

        return RouteDecision(
            model=chosen_model,
            confidence=round(float(probs[chosen_idx]), 4),
            scores=scores,
            strategy="cma-es",
            latency_ms=round(elapsed, 2),
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "generation": self._generation,
            "best_fitness": round(self.best_fitness, 4) if self.best_fitness > -float("inf") else 0.0,
            "sigma": round(self.sigma, 4),
            "n_models": self.n_models,
            "model_names": [n.split("/")[-1].split(":")[0] for n in self.model_names],
            "history_length": len(self._history),
        }

    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.data_dir, "cmaes_params.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "mean": self.mean.tolist(),
            "sigma": float(self.sigma),
            "best_fitness": float(self.best_fitness),
            "best_params": self.best_params.tolist() if self.best_params is not None else None,
            "generation": self._generation,
            "model_names": self.model_names,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.data_dir, "cmaes_params.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.mean = np.array(data["mean"])
        self.sigma = data["sigma"]
        self.best_fitness = data["best_fitness"]
        self.best_params = np.array(data["best_params"]) if data.get("best_params") else None
        self._generation = data.get("generation", 0)
        self.model_names = data.get("model_names", self.model_names)
        self.n_models = len(self.model_names)
