"""
Learned router — ModernBERT classifier fine-tuned on (prompt, model) pairs.

One forward pass through ModernBERT → [CLS] → classifier head → softmax → best model.
Falls back to CostRouter if model not trained or confidence below threshold.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from .interface import BaseRouter, RouterResult

BACKBONE = "answerdotai/ModernBERT-base"


class LearnedRouter(BaseRouter):
    """ModernBERT sequence classifier trained to pick the best model for a prompt."""

    name = "learned"

    def __init__(
        self,
        model_dir: str = ".fugusashi_data/router_model",
        fallback_strategy: str = "cost",
    ):
        self.model_dir = model_dir
        self.fallback_strategy = fallback_strategy
        self._model = None
        self._tokenizer = None
        self._model_names: List[str] = []
        self._loaded = False

    @property
    def is_trained(self) -> bool:
        config_path = os.path.join(self.model_dir, "config.json")
        names_path = os.path.join(self.model_dir, "model_names.json")
        return os.path.exists(config_path) and os.path.exists(names_path)

    def _load(self):
        if self._loaded:
            return
        if not self.is_trained:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        config = AutoConfig.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, config=config
        )
        self._model.eval()

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model.to(device)
        self._device = device

        with open(os.path.join(self.model_dir, "model_names.json")) as f:
            self._model_names = json.load(f)

        self._loaded = True

    def route(
        self,
        prompt: str,
        messages: List[Dict[str, str]],
        available_models: Dict[str, dict],
        threshold: float = 0.0,
    ) -> RouterResult:
        if not self.is_trained:
            return self._fallback(prompt, messages, available_models, threshold)

        self._load()

        start = time.perf_counter()
        inputs = self._tokenizer(
            prompt,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]

        confidence = float(probs.max().item())
        elapsed_ms = (time.perf_counter() - start) * 1000

        scores = {
            self._model_names[i]: float(probs[i].item())
            for i in range(len(self._model_names))
        }

        filtered_scores = {
            m: s for m, s in scores.items() if m in available_models
        }

        if not filtered_scores:
            return self._fallback(prompt, messages, available_models, threshold)

        if confidence < threshold:
            return self._fallback(prompt, messages, available_models, threshold)

        best_model = max(filtered_scores, key=filtered_scores.get)

        return RouterResult(
            model=best_model,
            confidence=min(confidence, 0.99),
            latency_ms=round(elapsed_ms, 2),
            strategy="learned",
            scores=filtered_scores,
            explanation=(
                f"ModernBERT classifier predicted {best_model} "
                f"(confidence: {confidence:.2f})"
            ),
            needs_escalation=False,
        )

    def _fallback(
        self,
        prompt: str,
        messages: List[Dict[str, str]],
        available_models: Dict[str, dict],
        threshold: float,
    ) -> RouterResult:
        from .strategies import CostRouter
        return CostRouter().route(prompt, messages, available_models, threshold)
