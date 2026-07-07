from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fugusashi.orchestrator import (
    MultiAgentOrchestrator,
    SubTask,
    TaskPlan,
    TaskStatus,
    TaskType,
    _classify_prompt,
    _best_model_for_type,
)
from fugusashi.grpo import GRPOTrainer, RoutingPolicy, TeamReward
from fugusashi.providers import ModelClient


def _make_mock_client():
    mock = MagicMock(spec=ModelClient)
    mock.get_available_models.return_value = {
        "gpt-4o-mini": {
            "name": "gpt-4o-mini",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "capabilities": ["chat", "reasoning", "code", "creative"],
            "cost_per_input_token": 0.00000015,
            "cost_per_output_token": 0.0000006,
        },
        "llama3.2-local": {
            "name": "llama3.2-local",
            "provider": "ollama",
            "model": "llama3.2:1b",
            "capabilities": ["chat", "reasoning"],
            "cost_per_input_token": 0.0,
            "cost_per_output_token": 0.0,
        },
    }
    mock.model_configs = mock.get_available_models.return_value
    return mock


class TestClassifyPrompt:
    def test_code_prompt(self):
        assert _classify_prompt("Write a Python function to sort a list") == TaskType.CODE

    def test_reasoning_prompt(self):
        assert _classify_prompt("Why is the sky blue?") == TaskType.REASONING

    def test_creative_prompt(self):
        result = _classify_prompt("Write a poem about the ocean")
        assert result in (TaskType.CREATIVE, TaskType.CODE)

    def test_factual_prompt(self):
        result = _classify_prompt("What is the capital of France?")
        assert result in (TaskType.FACTUAL, TaskType.CODE)

    def test_unknown_prompt(self):
        result = _classify_prompt("Hello there")
        assert result in (TaskType.REASONING, TaskType.FACTUAL)


class TestBestModelForType:
    def test_code_type(self):
        models = {
            "gpt-4o-mini": {"capabilities": ["chat", "reasoning", "code"]},
            "llama3.2-local": {"capabilities": ["chat", "reasoning"]},
        }
        result = _best_model_for_type(TaskType.CODE, models)
        assert result == "gpt-4o-mini"

    def test_empty_models(self):
        result = _best_model_for_type(TaskType.CODE, {})
        assert result is None


class TestOrchestrator:
    def test_rule_based_decompose(self):
        client = _make_mock_client()
        orch = MultiAgentOrchestrator(model_client=client)

        plan = TaskPlan(id="test", original_prompt="Write code")
        result = orch._rule_based_decompose("Write a Python function", plan)
        assert len(result.subtasks) == 1
        assert result.subtasks[0].task_type == TaskType.CODE

    def test_assign_models(self):
        client = _make_mock_client()
        orch = MultiAgentOrchestrator(model_client=client)

        plan = TaskPlan(
            id="test",
            original_prompt="test",
            subtasks=[
                SubTask(id="sub-0", description="code task", task_type=TaskType.CODE),
                SubTask(id="sub-1", description="creative task", task_type=TaskType.CREATIVE),
            ],
        )
        import asyncio
        asyncio.run(orch._assign_models(plan))

        assert plan.subtasks[0].assigned_model is not None
        assert plan.subtasks[1].assigned_model is not None
        assert plan.subtasks[0].status == TaskStatus.ASSIGNED

    def test_build_context(self):
        client = _make_mock_client()
        orch = MultiAgentOrchestrator(model_client=client)

        st = SubTask(id="sub-1", description="synthesis task", task_type=TaskType.SYNTHESIS, depends_on=["sub-0"])
        completed = {
            "sub-0": SubTask(
                id="sub-0", description="code task", task_type=TaskType.CODE,
                result="def sort(): pass", status=TaskStatus.COMPLETED,
            )
        }
        ctx = orch._build_context(st, completed)
        assert "synthesis" in ctx.lower()
        assert "def sort" in ctx

    def test_explain(self):
        client = _make_mock_client()
        orch = MultiAgentOrchestrator(model_client=client)

        plan = TaskPlan(
            id="test",
            original_prompt="test",
            subtasks=[
                SubTask(id="sub-0", description="task", task_type=TaskType.CODE,
                        assigned_model="gpt-4o-mini", latency_ms=100, status=TaskStatus.COMPLETED),
            ],
        )
        explanation = orch._explain(plan)
        assert "1 subtask" in explanation
        assert "gpt-4o-mini" in explanation


class TestGRPO:
    def test_scoring(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            trainer = GRPOTrainer(data_path=path)
            plan = TaskPlan(
                id="test",
                original_prompt="test",
                subtasks=[
                    SubTask(id="sub-0", description="task", task_type=TaskType.CODE,
                            assigned_model="gpt-4o-mini", status=TaskStatus.COMPLETED),
                ],
            )
            from fugusashi.orchestrator import OrchestratorResult
            result = OrchestratorResult(
                request_id="test",
                plan=plan,
                final_response="done",
                total_latency_ms=100,
                total_cost=0.001,
                models_used=["gpt-4o-mini"],
                explanation="test",
            )
            reward = trainer.score(result)
            assert 0 <= reward.reward <= 1
            assert reward.decomposition_score == 1.0
        finally:
            os.unlink(path)

    def test_policy_update(self):
        policy = RoutingPolicy(learning_rate=0.1)
        policy.type_model_probs = {"code": {"gpt-4o-mini": 0.5, "llama": 0.5}}
        policy.update("code", "gpt-4o-mini", reward=0.9, baseline=0.5)
        probs = policy.type_model_probs["code"]
        assert probs["gpt-4o-mini"] > 0.5

    def test_policy_select(self):
        import random
        policy = RoutingPolicy(learning_rate=0.1, exploration_rate=0.0)
        policy.type_model_probs = {"code": {"gpt-4o-mini": 0.9, "llama": 0.1}}
        random.seed(0)
        selected = policy.select("code", ["gpt-4o-mini", "llama"])
        assert selected in ("gpt-4o-mini", "llama")

    def test_grpo_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            trainer1 = GRPOTrainer(data_path=path, learning_rate=0.1)
            trainer1.baseline = 0.7
            trainer1.policy.type_model_probs = {"code": {"gpt-4o-mini": 0.8}}
            trainer1._save()

            trainer2 = GRPOTrainer(data_path=path)
            assert trainer2.baseline == 0.7
            assert trainer2.policy.type_model_probs == {"code": {"gpt-4o-mini": 0.8}}
        finally:
            os.unlink(path)

    def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            trainer = GRPOTrainer(data_path=path)
            stats = trainer.get_stats()
            assert stats["total_runs"] == 0
            assert stats["avg_reward"] == 0
        finally:
            os.unlink(path)


class TestTaskPlan:
    def test_to_dict(self):
        plan = TaskPlan(
            id="test",
            original_prompt="hello",
            subtasks=[
                SubTask(id="sub-0", description="task", task_type=TaskType.CODE),
            ],
            status=TaskStatus.COMPLETED,
            final_result="done",
        )
        d = plan.to_dict()
        assert d["id"] == "test"
        assert len(d["subtasks"]) == 1
        assert d["subtasks"][0]["task_type"] == "code"
