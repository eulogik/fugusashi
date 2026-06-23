from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from .router import EnsembleRouter


@dataclass
class BenchmarkSample:
    prompt: str
    expected_model: str
    category: str = "general"
    ideal_cost: float = 0.0


@dataclass
class BenchmarkResult:
    sample: BenchmarkSample
    chosen_model: str
    confidence: float
    strategy: str
    correct: bool
    routing_latency_ms: float
    cost_if_routed: float
    cost_if_ideal: float


@dataclass
class BenchmarkReport:
    results: List[BenchmarkResult] = field(default_factory=list)
    router: Optional[EnsembleRouter] = None

    def add(self, r: BenchmarkResult):
        self.results.append(r)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)

    @property
    def cost_savings_pct(self) -> float:
        if not self.results:
            return 0.0
        routed = sum(r.cost_if_routed for r in self.results)
        ideal = sum(r.cost_if_ideal for r in self.results)
        if ideal == 0:
            return 0.0
        return (1 - routed / ideal) * 100

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.routing_latency_ms for r in self.results) / len(self.results)

    @property
    def strategy_distribution(self) -> Dict[str, int]:
        dist = {}
        for r in self.results:
            dist[r.strategy] = dist.get(r.strategy, 0) + 1
        return dist

    @property
    def model_distribution(self) -> Dict[str, int]:
        dist = {}
        for r in self.results:
            dist[r.chosen_model] = dist.get(r.chosen_model, 0) + 1
        return dist

    @property
    def by_category(self) -> Dict[str, Dict[str, float]]:
        cats = {}
        for r in self.results:
            cat = r.sample.category
            if cat not in cats:
                cats[cat] = {"total": 0, "correct": 0, "cost_routed": 0.0, "cost_ideal": 0.0}
            cats[cat]["total"] += 1
            cats[cat]["correct"] += 1 if r.correct else 0
            cats[cat]["cost_routed"] += r.cost_if_routed
            cats[cat]["cost_ideal"] += r.cost_if_ideal
        return cats

    def print_report(self):
        click.echo("\n" + "=" * 60)
        click.echo("  FUGUSASHI BENCHMARK REPORT")
        click.echo("=" * 60)

        click.echo(f"\n  Total samples:    {len(self.results)}")
        click.echo(f"  Routing accuracy: {self.accuracy:.1%}")
        click.echo(f"  Cost vs ideal:    {self.cost_savings_pct:+.1f}%")
        click.echo(f"  Avg routing time: {self.avg_latency_ms:.2f}ms")

        click.echo(f"\n  ── Strategy Distribution ──")
        for strat, count in sorted(self.strategy_distribution.items(), key=lambda x: -x[1]):
            pct = count / len(self.results) * 100
            click.echo(f"    {strat:30s} {count:3d} ({pct:.0f}%)")

        click.echo(f"\n  ── Model Distribution ──")
        for model, count in sorted(self.model_distribution.items(), key=lambda x: -x[1]):
            pct = count / len(self.results) * 100
            click.echo(f"    {model:30s} {count:3d} ({pct:.0f}%)")

        click.echo(f"\n  ── By Category ──")
        for cat, data in sorted(self.by_category.items()):
            acc = data["correct"] / data["total"] if data["total"] else 0
            savings = (1 - data["cost_routed"] / data["cost_ideal"]) * 100 if data["cost_ideal"] else 0
            click.echo(f"    {cat:20s} acc={acc:.0%}  cost_savings={savings:+.0f}%  n={data['total']}")

        click.echo("\n  ── Sample Mismatches (first 10) ──")
        mismatches = [r for r in self.results if not r.correct][:10]
        if mismatches:
            for r in mismatches:
                click.echo(
                    f"    expected={r.sample.expected_model:20s} "
                    f"got={r.chosen_model:20s} "
                    f"conf={r.confidence:.2f} "
                    f"strat={r.strategy}"
                )
        else:
            click.echo("    (none)")

        click.echo("=" * 60 + "\n")

    def to_json(self) -> dict:
        return {
            "samples": len(self.results),
            "accuracy": self.accuracy,
            "cost_savings_pct": self.cost_savings_pct,
            "avg_latency_ms": self.avg_latency_ms,
            "strategy_distribution": self.strategy_distribution,
            "model_distribution": self.model_distribution,
            "by_category": self.by_category,
        }


DEFAULT_DATASET = [
    BenchmarkSample(prompt="Write a Python function to sort a list", expected_model="llama3.2-local", category="code", ideal_cost=0.0),
    BenchmarkSample(prompt="What is the capital of France?", expected_model="llama3.2-local", category="factual", ideal_cost=0.0),
    BenchmarkSample(prompt="Write a poem about autumn", expected_model="llama3.2-local", category="creative", ideal_cost=0.0),
    BenchmarkSample(prompt="Debug this: console.log('hello'", expected_model="gpt-4o-mini", category="code", ideal_cost=0.00001),
    BenchmarkSample(prompt="Implement merge sort in Python", expected_model="gpt-4o-mini", category="code", ideal_cost=0.00002),
    BenchmarkSample(prompt="What is 2 + 2?", expected_model="llama3.2-local", category="factual", ideal_cost=0.0),
    BenchmarkSample(prompt="Write a bash script to backup a directory", expected_model="gpt-4o-mini", category="code", ideal_cost=0.00001),
    BenchmarkSample(prompt="Describe the water cycle", expected_model="llama3.2-local", category="factual", ideal_cost=0.0),
    BenchmarkSample(prompt="Explain how HTTP works", expected_model="llama3.2-local", category="explanation", ideal_cost=0.0),
    BenchmarkSample(prompt="Write a SQL query to join two tables", expected_model="gpt-4o-mini", category="code", ideal_cost=0.00001),
    BenchmarkSample(prompt="What is the meaning of life?", expected_model="llama3.2-local", category="general", ideal_cost=0.0),
    BenchmarkSample(prompt="Create a React component for a todo list", expected_model="gpt-4o-mini", category="code", ideal_cost=0.00002),
    BenchmarkSample(prompt="Summarize the plot of The Great Gatsby", expected_model="llama3.2-local", category="general", ideal_cost=0.0),
    BenchmarkSample(prompt="Write a Dockerfile for a Node.js app", expected_model="gpt-4o-mini", category="code", ideal_cost=0.00001),
    BenchmarkSample(prompt="Convert this Python code to JavaScript", expected_model="gpt-4o-mini", category="code", ideal_cost=0.00001),
    BenchmarkSample(prompt="Tell me a joke", expected_model="llama3.2-local", category="creative", ideal_cost=0.0),
    BenchmarkSample(prompt="What is the Pythagorean theorem?", expected_model="llama3.2-local", category="factual", ideal_cost=0.0),
    BenchmarkSample(prompt="Write a regular expression for email validation", expected_model="gpt-4o-mini", category="code", ideal_cost=0.00001),
    BenchmarkSample(prompt="Explain the concept of recursion", expected_model="llama3.2-local", category="code", ideal_cost=0.0),
    BenchmarkSample(prompt="Write a poem about AI", expected_model="llama3.2-local", category="creative", ideal_cost=0.0),
]


def run_benchmark(
    router: EnsembleRouter,
    dataset: List[BenchmarkSample],
    models_config: Dict[str, dict],
    confidence_threshold: float = 0.4,
    verbose: bool = False,
) -> BenchmarkReport:
    report = BenchmarkReport(router=router)

    for sample in dataset:
        start = time.perf_counter()
        result = router.route(
            prompt=sample.prompt,
            messages=[{"role": "user", "content": sample.prompt}],
            available_models=models_config,
            threshold=confidence_threshold,
        )
        elapsed = (time.perf_counter() - start) * 1000

        chosen_model = result.model
        correct = chosen_model == sample.expected_model

        cfg = models_config.get(chosen_model, {})
        cost_routed = cfg.get("cost_per_input_token", 0) * 50 + cfg.get("cost_per_output_token", 0) * 100

        ideal_cfg = models_config.get(sample.expected_model, {})
        cost_ideal = ideal_cfg.get("cost_per_input_token", 0) * 50 + ideal_cfg.get("cost_per_output_token", 0) * 100

        br = BenchmarkResult(
            sample=sample,
            chosen_model=chosen_model,
            confidence=result.confidence,
            strategy=result.strategy,
            correct=correct,
            routing_latency_ms=elapsed,
            cost_if_routed=cost_routed,
            cost_if_ideal=cost_ideal,
        )
        report.add(br)

        if verbose:
            icon = "✓" if correct else "✗"
            click.echo(f"  {icon} {sample.prompt[:50]:50s} -> {chosen_model:20s} (exp={sample.expected_model:20s}) [{result.strategy}]")

    return report


DEFAULT_TRAINING = [
    {"prompt": "Write a Python function to sort a list", "model": "gpt-4o-mini", "score": 0.95},
    {"prompt": "Debug a JavaScript console.log statement", "model": "gpt-4o-mini", "score": 0.9},
    {"prompt": "Write a bash script to backup files", "model": "gpt-4o-mini", "score": 0.85},
    {"prompt": "Write a SQL query with JOIN", "model": "gpt-4o-mini", "score": 0.9},
    {"prompt": "Create a React component", "model": "gpt-4o-mini", "score": 0.9},
    {"prompt": "Write a Dockerfile for a web app", "model": "gpt-4o-mini", "score": 0.85},
    {"prompt": "Convert code between languages", "model": "gpt-4o-mini", "score": 0.9},
    {"prompt": "Write a regex pattern for validation", "model": "gpt-4o-mini", "score": 0.85},
]


def run_benchmark_cli(dataset_path=None, threshold=0.4, verbose=False, json_out=False, train=False):
    from .router import EnsembleRouter

    router = EnsembleRouter(confidence_threshold=threshold)

    if train:
        router.similarity_router.build_index(DEFAULT_TRAINING)

    models_config = {
        "llama3.2-local": {
            "cost_per_input_token": 0.0, "cost_per_output_token": 0.0,
            "capabilities": ["chat"],
        },
        "gpt-4o-mini": {
            "cost_per_input_token": 0.00000015, "cost_per_output_token": 0.0000006,
            "capabilities": ["chat", "code", "reasoning"],
        },
        "gpt-4o": {
            "cost_per_input_token": 0.0000025, "cost_per_output_token": 0.00001,
            "capabilities": ["chat", "code", "reasoning"],
        },
    }

    if dataset_path:
        samples = []
        with open(dataset_path) as f:
            for line in f:
                data = json.loads(line)
                samples.append(BenchmarkSample(
                    prompt=data["prompt"],
                    expected_model=data["expected_model"],
                    category=data.get("category", "general"),
                    ideal_cost=data.get("ideal_cost", 0.0),
                ))
    else:
        click.echo("Using default benchmark dataset (20 samples)")
        samples = DEFAULT_DATASET

    if train:
        click.echo("Seeded similarity router with 8 training examples")

    if verbose:
        click.echo(f"Running benchmark on {len(samples)} samples...\n")

    report = run_benchmark(router, samples, models_config, threshold, verbose=verbose)

    if json_out:
        click.echo(json.dumps(report.to_json(), indent=2))
    else:
        report.print_report()

    return report


@click.command()
@click.option("--dataset", "-d", type=click.Path(exists=True), help="JSONL dataset file")
@click.option("--threshold", "-t", default=0.4, type=float, help="Confidence threshold")
@click.option("--verbose", "-v", is_flag=True, help="Show per-sample results")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
@click.option("--train", is_flag=True, help="Seed training data for similarity routing")
def benchmark(dataset, threshold, verbose, json_out, train):
    run_benchmark_cli(dataset_path=dataset, threshold=threshold, verbose=verbose, json_out=json_out, train=train)


if __name__ == "__main__":
    benchmark()
