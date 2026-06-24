"""
Community preference dataset for Fugusashi.

Sources:
- Chatbot Arena (LMSYS): https://huggingface.co/datasets/lmsys/chatbot_arena_conversations
- MT-Bench: https://huggingface.co/datasets/lmsys/mt_bench

This module provides tools to import, export, and manage community-curated
prompt → model preference data for training the Fugusashi router.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Preference:
    prompt: str
    preferred_model: str
    source: str = "community"
    category: str = "general"
    score: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PreferenceDataset:
    def __init__(self, data_dir: str = ".fugusashi_data"):
        self.data_dir = data_dir
        self.preferences: List[Preference] = []
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(self.data_dir, exist_ok=True)

    def add(self, prompt: str, model: str, source: str = "community",
            category: str = "general", score: float = 1.0,
            metadata: Optional[Dict[str, Any]] = None):
        self.preferences.append(Preference(
            prompt=prompt,
            preferred_model=model,
            source=source,
            category=category,
            score=score,
            metadata=metadata or {},
        ))

    def save(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.data_dir, "preferences.jsonl")
        with open(path, "w") as f:
            for p in self.preferences:
                f.write(json.dumps({
                    "prompt": p.prompt,
                    "preferred_model": p.preferred_model,
                    "source": p.source,
                    "category": p.category,
                    "score": p.score,
                    "metadata": p.metadata,
                }) + "\n")
        return len(self.preferences)

    def load(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(self.data_dir, "preferences.jsonl")
        if not os.path.exists(path):
            return 0
        count = 0
        with open(path) as f:
            for line in f:
                data = json.loads(line)
                self.preferences.append(Preference(**data))
                count += 1
        return count

    def export_for_training(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        if path is None:
            path = os.path.join(self.data_dir, "training_data.jsonl")
        data = []
        for p in self.preferences:
            entry = {
                "prompt": p.prompt,
                "model": p.preferred_model,
                "score": p.score,
            }
            data.append(entry)
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        return data

    def get_stats(self) -> Dict[str, Any]:
        sources = {}
        categories = {}
        models = {}
        for p in self.preferences:
            sources[p.source] = sources.get(p.source, 0) + 1
            categories[p.category] = categories.get(p.category, 0) + 1
            models[p.preferred_model] = models.get(p.preferred_model, 0) + 1
        return {
            "total": len(self.preferences),
            "sources": sources,
            "categories": categories,
            "models": models,
        }


def seed_default_dataset(dataset: PreferenceDataset) -> int:
    defaults = [
        ("Write a Python function to sort a list", "openai/gpt-oss-120b:free", "code", 0.95),
        ("What is the capital of France?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("Explain quantum entanglement simply", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("Write a poem about AI", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
        ("Debug this JavaScript error", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Write a bash script to backup files", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Explain the theory of relativity", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("Write a SQL query to join tables", "openai/gpt-oss-120b:free", "code", 0.9),
        ("What is the meaning of life?", "liquid/lfm-2.5-1.2b-instruct:free", "general", 0.6),
        ("Write a recursive Fibonacci function", "openai/gpt-oss-120b:free", "code", 0.95),
        ("What is the capital of Japan?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("Create a React component for a todo list", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Summarize the plot of The Great Gatsby", "liquid/lfm-2.5-1.2b-instruct:free", "general", 0.7),
        ("Write a Dockerfile for a Node.js app", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Convert Python code to JavaScript", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Tell me a joke", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.6),
        ("What is the Pythagorean theorem?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("Write a regex for email validation", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Explain recursion with examples", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Write a poem about autumn", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
    ]
    for prompt, model, category, score in defaults:
        dataset.add(prompt, model, source="eulogik-seed", category=category, score=score)
    return len(defaults)
