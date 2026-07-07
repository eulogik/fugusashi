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
    """Seed 3-model-class dataset: oss-120b (code/reasoning), hermes-3 (medium), lfm (simple)."""
    defaults = [
        # --- code / reasoning → gpt-oss-120b ---
        ("Write a Python function to sort a list", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Implement merge sort in Java", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Write a REST API endpoint in FastAPI", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Debug this JavaScript error: 'undefined is not a function'", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Write a recursive Fibonacci function", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Create a React component for a todo list", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a Dockerfile for a Node.js app", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Write a SQL query to join three tables", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a regex for email validation", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Explain recursion with examples in Python", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Implement a binary search in Python", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Write unit tests for a login function", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a Python decorator for caching", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Implement a hash map from scratch", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Write a Python async function that fetches multiple URLs", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Create a CI/CD pipeline YAML for GitHub Actions", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Write a Go channel example with worker pool", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Implement a LRU cache in Python", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a Python context manager for timing code", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Create a SQL migration script for adding columns", "openai/gpt-oss-120b:free", "code", 0.85),
        # --- complex explanation / medium → hermes-3-405b ---
        ("Explain the theory of general relativity simply", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does the event loop work in Node.js?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is blockchain technology?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain the CAP theorem", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        ("How does encryption work?", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        ("What is quantum computing?", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        ("Explain how neural networks learn", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        ("How does TCP/IP work?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is the difference between TCP and UDP?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does a compiler work?", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        ("What is CRISPR gene editing?", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        ("Explain the difference between AI and ML", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does a search engine index the web?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain the concept of sharding in databases", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        ("How does OAuth2 authentication work?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is a microservice architecture?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does a CDN work?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is the difference between REST and GraphQL?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain how containerization differs from VMs", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does garbage collection work in Python?", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        # Additional hermes-3 examples for better differentiation ↑
        ("Explain how DNS resolves domain names", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How do load balancers distribute traffic?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is the OSI model?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain how WebSockets maintain persistent connections", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does a database index speed up queries?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is the difference between authorization and authentication?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain how VPNs create secure tunnels", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does Docker container networking work?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is a message queue and when to use it?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain the difference between process and thread", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does SSL/TLS establish trust?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is the role of a reverse proxy?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain how HTTP cookies work", "meta-llama/hermes-3-405b:free", "explanation", 0.75),
        ("How does RAID storage improve reliability?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is the difference between Git merge and rebase?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain how CDNs cache and serve content", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How do API gateways manage microservices?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is the difference between symmetric and asymmetric encryption?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain how blue-green deployments work", "meta-llama/hermes-3-405b:free", "explanation", 0.75),
        ("How does circuit breaker pattern prevent cascade failures?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        # --- factual / simple → lfm-2.5-1.2b ---
        ("What is the capital of France?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is the capital of Japan?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is the meaning of life?", "liquid/lfm-2.5-1.2b-instruct:free", "general", 0.6),
        ("What is the Pythagorean theorem?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is photosynthesis?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("Who invented the telephone?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is the speed of light?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("When was the US Declaration of Independence signed?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is the atomic number of carbon?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("How many planets are in the solar system?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is the largest ocean on Earth?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("Who wrote Romeo and Juliet?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is the boiling point of water?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What year did World War II end?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is the tallest mountain in the world?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("Who painted the Mona Lisa?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is the powerhouse of the cell?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is the hardest natural substance?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("How many bones are in the human body?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        ("What is the chemical formula for gold?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.85),
        # --- creative → lfm-2.5-1.2b ---
        ("Write a haiku about programming", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.65),
        ("Tell me a joke", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.6),
        ("Write a poem about autumn", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.65),
        ("Summarize the plot of The Great Gatsby", "liquid/lfm-2.5-1.2b-instruct:free", "general", 0.7),
        ("Describe a sunset in the style of Shakespeare", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.65),
        ("Write a limerick about a programmer", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.65),
        ("Write a short story about a robot learning to paint", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.65),
        ("Create a metaphor for artificial intelligence", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.65),
        ("Write a dialogue between two AI assistants", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.65),
        ("Describe what happiness feels like", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.6),
    ]
    for prompt, model, category, score in defaults:
        dataset.add(prompt, model, source="eulogik-seed", category=category, score=score)
    return len(defaults)
