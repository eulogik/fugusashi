"""
Training pipeline for the Fugusashi learned router.

Fine-tunes a ModernBERT classifier on (prompt, model) pairs to predict the
best model for each prompt. One forward pass — no cross-encoder overhead.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


BACKBONE = "answerdotai/ModernBERT-base"


@dataclass
class TrainingConfig:
    epochs: int = 6
    learning_rate: float = 5e-5
    batch_size: int = 8
    test_split: float = 0.2
    min_samples: int = 20
    warmup_ratio: float = 0.1
    max_length: int = 96
    patience: int = 3


@dataclass
class TrainingResult:
    accuracy: float
    top3_accuracy: float
    cost_savings: float
    epochs_trained: int
    training_time_ms: float
    model_path: str
    n_classes: int
    n_train: int
    n_test: int
    per_class_accuracy: Dict[str, float]
    backbone: str

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "top3_accuracy": self.top3_accuracy,
            "cost_savings": self.cost_savings,
            "epochs_trained": self.epochs_trained,
            "training_time_ms": self.training_time_ms,
            "model_path": self.model_path,
            "n_classes": self.n_classes,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "per_class_accuracy": self.per_class_accuracy,
            "backbone": self.backbone,
        }


def load_dataset(data_dir: str = ".fugusashi_data") -> List[Dict[str, Any]]:
    """Load all preference data from JSONL files, including feedback outcomes."""
    samples = []
    for fname in ["preferences.jsonl", "training_data.jsonl", "expanded_preferences.jsonl", "outcomes.jsonl"]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            samples.append(data)
                        except json.JSONDecodeError:
                            continue

    # Convert feedback outcomes to training format
    converted = []
    for s in samples:
        if "routed_to" in s and "prompt" in s:
            score = s.get("auto_score", 0.5)
            if s.get("user_rating") is not None:
                score = s["user_rating"] / 5.0
            if s.get("error", False):
                score = 0.0
            converted.append({
                "prompt": s["prompt"],
                "model": s["routed_to"],
                "preferred_model": s["routed_to"],
                "score": max(0.0, min(2.0, score)),
                "source": "feedback",
                "category": s.get("category", "general"),
            })
    samples.extend(converted)

    return samples


def train_modernbert(
    model_dir: str = ".fugusashi_data/router_model",
    data_dir: str = ".fugusashi_data",
    config: Optional[TrainingConfig] = None,
) -> TrainingResult:
    """Fine-tune a ModernBERT sequence classifier on (prompt, model) pairs."""
    if config is None:
        config = TrainingConfig()

    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_scheduler,
    )
    import torch
    from torch.utils.data import DataLoader, Dataset

    samples = load_dataset(data_dir)
    if not samples:
        raise ValueError(
            f"No training data found in {data_dir}. "
            "Run 'fugusashi expand-data' first."
        )

    model_names = sorted(set(s.get("model", s.get("preferred_model", "")) for s in samples))
    model_to_idx = {m: i for i, m in enumerate(model_names)}
    n_classes = len(model_names)

    if n_classes < 2:
        raise ValueError(f"Need at least 2 model classes, got {n_classes}")

    prompts, labels, weights = [], [], []
    for s in samples:
        prompt = s.get("prompt", "")
        model = s.get("model", s.get("preferred_model", ""))
        score = s.get("score", 1.0)
        if prompt and model in model_to_idx:
            prompts.append(prompt)
            labels.append(model_to_idx[model])
            weights.append(min(score, 2.0))

    if len(prompts) < config.min_samples:
        raise ValueError(
            f"Need at least {config.min_samples} samples, got {len(prompts)}. "
            "Run 'fugusashi expand-data' first."
        )

    import random
    combined = list(zip(prompts, labels, weights))
    random.shuffle(combined)
    prompts, labels, weights = zip(*combined)
    prompts = list(prompts)
    labels = np.array(labels, dtype=np.int64)
    weights_arr = np.array(weights, dtype=np.float32)

    split = int(len(prompts) * (1 - config.test_split))
    train_prompts, test_prompts = prompts[:split], prompts[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    train_weights, test_weights = weights_arr[:split], weights_arr[split:]

    start_time = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    config_model = AutoConfig.from_pretrained(
        BACKBONE, num_labels=n_classes, problem_type="single_label_classification"
    )
    model = AutoModelForSequenceClassification.from_pretrained(BACKBONE, config=config_model)

    class PromptDataset(Dataset):
        def __init__(self, prompts, labels, weights_):
            self.prompts = prompts
            self.labels = labels
            self.weights = weights_

        def __len__(self):
            return len(self.prompts)

        def __getitem__(self, idx):
            tokens = tokenizer(
                self.prompts[idx],
                truncation=True,
                max_length=config.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": tokens["input_ids"][0],
                "attention_mask": tokens["attention_mask"][0],
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
                "weight": torch.tensor(self.weights[idx], dtype=torch.float32),
            }

    train_dataset = PromptDataset(train_prompts, train_labels, train_weights)
    test_dataset = PromptDataset(test_prompts, test_labels, test_weights)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    num_training_steps = len(train_loader) * config.epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            batch_weights = batch["weight"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits, batch_labels, reduction="none")
            loss = (loss * batch_weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        correct, total = 0, 0
        top3_correct = 0
        per_class_correct: Dict[int, int] = {}
        per_class_total: Dict[int, int] = {}
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                preds = probs.argmax(dim=-1)
                top3_preds = probs.topk(k=min(3, n_classes), dim=-1).indices

                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)
                for i in range(len(batch_labels)):
                    top3_correct += 1 if batch_labels[i] in top3_preds[i] else 0
                    lbl = batch_labels[i].item()
                    per_class_total[lbl] = per_class_total.get(lbl, 0) + 1
                    per_class_correct[lbl] = per_class_correct.get(lbl, 0) + (
                        1 if preds[i] == batch_labels[i] else 0
                    )

        accuracy = correct / total if total > 0 else 0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    correct, total = 0, 0
    top3_correct = 0
    per_class_correct = {}
    per_class_total = {}
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)
            top3_preds = probs.topk(k=min(3, n_classes), dim=-1).indices

            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)
            for i in range(len(batch_labels)):
                top3_correct += 1 if batch_labels[i] in top3_preds[i] else 0
                lbl = batch_labels[i].item()
                per_class_total[lbl] = per_class_total.get(lbl, 0) + 1
                per_class_correct[lbl] = per_class_correct.get(lbl, 0) + (
                    1 if preds[i] == batch_labels[i] else 0
                )

    accuracy = correct / total if total > 0 else 0
    top3_accuracy = top3_correct / total if total > 0 else 0

    per_class_acc = {}
    for c in range(n_classes):
        if per_class_total.get(c, 0) > 0:
            per_class_acc[model_names[c]] = per_class_correct[c] / per_class_total[c]

    free_models = [m for m in model_names if ":free" in m.lower() or "local" in m.lower()]
    cost_savings = 0.0
    if free_models and total > 0:
        model.eval()
        free_preds = 0
        total_preds = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                for p in preds:
                    if model_names[p.item()] in free_models:
                        free_preds += 1
                    total_preds += 1
        cost_savings = (free_preds / total_preds) * 100 if total_preds > 0 else 0.0

    os.makedirs(model_dir, exist_ok=True)
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "model_names.json"), "w") as f:
        json.dump(model_names, f)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return TrainingResult(
        accuracy=float(accuracy),
        top3_accuracy=float(top3_accuracy),
        cost_savings=cost_savings,
        epochs_trained=epoch + 1,
        training_time_ms=elapsed_ms,
        model_path=model_dir,
        n_classes=n_classes,
        n_train=len(train_prompts),
        n_test=len(test_prompts),
        per_class_accuracy=per_class_acc,
        backbone=BACKBONE,
    )


def expand_dataset(data_dir: str = ".fugusashi_data") -> int:
    """Generate expanded training data from seed data + synthetic variants."""
    from .dataset import seed_default_dataset, PreferenceDataset

    ds = PreferenceDataset(data_dir=data_dir)
    seed_default_dataset(ds)
    ds.save()

    seed_data = [
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
        ("Implement a binary search in Python", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Write a REST API endpoint in FastAPI", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Create a SQL migration script", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write unit tests for a login function", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Debug a memory leak in Node.js", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Write a Python decorator for caching", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Implement a linked list in C", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a CSS grid layout for a dashboard", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Create a Git pre-commit hook", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a Python context manager", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Implement merge sort in Java", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a GraphQL resolver", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Create a Kubernetes deployment YAML", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Write a Python async function", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Implement a hash map from scratch", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Write a bash script for log rotation", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Create a React hook for API calls", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a Python dataclass schema", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Implement a LRU cache in Go", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a Docker compose file for microservices", "openai/gpt-oss-120b:free", "code", 0.85),
        ("What is photosynthesis?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("Who invented the telephone?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the speed of light?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("When was the Declaration of Independence signed?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the atomic number of carbon?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("How many planets are in the solar system?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the largest ocean on Earth?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("Who wrote Romeo and Juliet?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the boiling point of water?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What year did World War II end?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the currency of Japan?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("How far is the Moon from Earth?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the tallest mountain in the world?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the chemical formula for gold?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("Who painted the Mona Lisa?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the largest country by area?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the powerhouse of the cell?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("How many bones are in the human body?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the national animal of India?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("What is the hardest natural substance?", "liquid/lfm-2.5-1.2b-instruct:free", "factual", 0.8),
        ("How does the internet work?", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("Explain machine learning to a beginner", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("What is blockchain technology?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("How does DNA replication work?", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("Explain the CAP theorem simply", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("How does public key encryption work?", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("What is quantum computing and why does it matter?", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("Explain how neural networks learn from data", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("How does the TCP/IP protocol stack work?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("What is the theory of general relativity?", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("Explain the difference between TCP and UDP protocols", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("How does a compiler translate source code?", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("What is CRISPR gene editing technology?", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("Explain how GPS satellites determine location", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("What is dark matter made of?", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("Explain the key differences between AI and ML", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("How does a search engine rank web pages?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("What is protein folding and why is it important?", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("Explain how 5G technology improves on 4G", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("How does the Node.js event loop work?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("How does garbage collection manage memory?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("Create a metaphor for artificial intelligence", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
        ("Write a short story about a robot", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
        ("Describe a sunset in the style of Shakespeare", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
        ("Write a limerick about a programmer", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
        ("Create a tagline for a coffee shop", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.6),
        ("Write a sonnet about the ocean", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
        ("Describe what happiness feels like", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.6),
        ("Write a parody of a famous song about coding", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
        ("Create a fantasy world description", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
        ("Write a dialogue between two AI assistants", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
        ("Describe a futuristic city in 2050", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.7),
        ("Write a product description for a time machine", "liquid/lfm-2.5-1.2b-instruct:free", "creative", 0.6),
        ("How does the event loop work in Node.js?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("How does garbage collection work?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("Explain the difference between REST and gRPC", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("How does TLS handshake establish secure connection?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("What is the difference between TCP and UDP at transport layer?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("How does database sharding improve performance?", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("How does a CDN speed up content delivery?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("SQL vs NoSQL: which to choose?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("How do containers differ from virtual machines?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("How does OAuth authorization code flow work?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("What problem does microservice architecture solve?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("Explain PACELC theorem in distributed databases", "openai/gpt-oss-120b:free", "explanation", 0.9),
        ("How does blockchain achieve Byzantine fault tolerance?", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("Mutation testing vs migration testing differences", "openai/gpt-oss-120b:free", "explanation", 0.85),
        ("Write a JavaScript Promise.all implementation", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Create a Python script to parse CSV files", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Write a Shell script to monitor disk usage", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Implement a binary tree in TypeScript", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Create a Redis caching layer in Python", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a Ruby method to merge hashes", "liquid/lfm-2.5-1.2b-instruct:free", "code", 0.7),
        ("Create a simple PHP login system", "liquid/lfm-2.5-1.2b-instruct:free", "code", 0.7),
        ("Write a Swift function to fetch API data", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Implement a Rust iterator", "openai/gpt-oss-120b:free", "code", 0.95),
        ("Write a Go channel example", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Create a Kotlin coroutine example", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a C++ template function", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Write a batch file to automate backups", "liquid/lfm-2.5-1.2b-instruct:free", "code", 0.6),
        ("Create a Terraform configuration for S3 bucket", "openai/gpt-oss-120b:free", "code", 0.85),
        ("Write a CloudFormation template for EC2", "openai/gpt-oss-120b:free", "code", 0.9),
        ("Create a CI/CD pipeline YAML for GitLab", "openai/gpt-oss-120b:free", "code", 0.85),
        # --- medium explanation → hermes-3-405b ---
        ("How does the internet work?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain machine learning to a beginner", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is blockchain technology?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does DNA replication work?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain the CAP theorem in distributed systems", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        ("How does a compiler work?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is CRISPR gene editing?", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        ("Explain the difference between AI and ML", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does a search engine index the web?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain the concept of sharding in databases", "meta-llama/hermes-3-405b:free", "explanation", 0.85),
        ("How does OAuth2 authentication work?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is a microservice architecture?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("Explain how containerization differs from virtualization", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does SSL/TLS encryption work?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("What is the difference between REST and GraphQL?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
        ("How does garbage collection work?", "meta-llama/hermes-3-405b:free", "explanation", 0.8),
    ]

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "expanded_preferences.jsonl")
    count = 0
    with open(path, "w") as f:
        seen = set()
        for prompt, model, category, score in seed_data:
            if prompt in seen:
                continue
            seen.add(prompt)
            entry = {
                "prompt": prompt,
                "model": model,
                "preferred_model": model,
                "source": "eulogik-seed-v2",
                "category": category,
                "score": score,
            }
            f.write(json.dumps(entry) + "\n")
            count += 1

    return count
