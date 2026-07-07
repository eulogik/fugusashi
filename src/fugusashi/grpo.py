from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .orchestrator import OrchestratorResult, TaskType


@dataclass
class TeamReward:
    plan_id: str
    reward: float
    decomposition_score: float
    routing_score: float
    synthesis_score: float
    latency_penalty: float
    cost_penalty: float


@dataclass
class RoutingPolicy:
    """Probability distribution over model assignments per task type."""
    type_model_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    learning_rate: float = 0.1
    exploration_rate: float = 0.1

    def get_prob(self, task_type: str, model: str) -> float:
        return self.type_model_probs.get(task_type, {}).get(model, 1.0)

    def update(
        self,
        task_type: str,
        model: str,
        reward: float,
        baseline: float = 0.5,
    ) -> None:
        if task_type not in self.type_model_probs:
            self.type_model_probs[task_type] = {}
        probs = self.type_model_probs[task_type]
        advantage = reward - baseline
        old_p = probs.get(model, 0.5)
        new_p = old_p + self.learning_rate * advantage * old_p * (1 - old_p)
        new_p = max(0.05, min(0.95, new_p))
        probs[model] = new_p
        total = sum(probs.values())
        if total > 0:
            for m in probs:
                probs[m] /= total

    def select(self, task_type: str, models: List[str]) -> str:
        import random
        if random.random() < self.exploration_rate:
            return random.choice(models)
        probs = self.type_model_probs.get(task_type, {})
        if not probs:
            return models[0] if models else ""
        weighted = [(m, probs.get(m, 0.0)) for m in models]
        total = sum(p for _, p in weighted)
        if total <= 0:
            return models[0]
        r = random.random() * total
        cumulative = 0.0
        for m, p in weighted:
            cumulative += p
            if r <= cumulative:
                return m
        return models[-1]


class GRPOTrainer:
    """Group Relative Policy Optimization for multi-agent teamwork.

    Learns which decomposition strategies and model assignments produce
    the best outcomes across tasks.
    """

    def __init__(
        self,
        data_path: str = ".fugusashi_data/grpo_state.json",
        learning_rate: float = 0.1,
        baseline_decay: float = 0.95,
    ):
        self.data_path = data_path
        self.baseline_decay = baseline_decay
        self.policy = RoutingPolicy(learning_rate=learning_rate)
        self.baseline: float = 0.5
        self.reward_history: List[TeamReward] = []
        self._load()

    def score(
        self,
        result: OrchestratorResult,
        expected_type: Optional[TaskType] = None,
    ) -> TeamReward:
        plan = result.plan
        n_completed = sum(
            1 for st in plan.subtasks if st.status.value == "completed"
        )
        n_total = len(plan.subtasks)
        if n_total == 0:
            return TeamReward(
                plan_id=plan.id,
                reward=0.0,
                decomposition_score=0.0,
                routing_score=0.0,
                synthesis_score=0.0,
                latency_penalty=0.0,
                cost_penalty=0.0,
            )

        decomposition_score = min(1.0, n_completed / n_total)

        routing_score = 0.0
        if expected_type:
            for st in plan.subtasks:
                if st.task_type == expected_type:
                    routing_score += 1.0
            routing_score /= n_total
        else:
            routing_score = decomposition_score

        has_synthesis = any(
            st.task_type == TaskType.SYNTHESIS for st in plan.subtasks
        )
        synthesis_score = 1.0 if (has_synthesis and n_total > 1) else (
            0.8 if n_total == 1 else 0.5
        )

        latency_s = result.total_latency_ms / 1000.0
        latency_penalty = 1.0 / (1.0 + math.exp(latency_s - 5.0))

        cost_score = 1.0 / (1.0 + result.total_cost * 100)

        reward = (
            decomposition_score * 0.3
            + routing_score * 0.3
            + synthesis_score * 0.2
            + latency_penalty * 0.1
            + cost_score * 0.1
        )

        team_reward = TeamReward(
            plan_id=plan.id,
            reward=reward,
            decomposition_score=decomposition_score,
            routing_score=routing_score,
            synthesis_score=synthesis_score,
            latency_penalty=latency_penalty,
            cost_penalty=cost_score,
        )
        self.reward_history.append(team_reward)
        return team_reward

    def update_policy(self, result: OrchestratorResult, reward: float) -> None:
        for st in result.plan.subtasks:
            if st.assigned_model:
                self.policy.update(
                    st.task_type.value,
                    st.assigned_model,
                    reward,
                    self.baseline,
                )
        self.baseline = (
            self.baseline_decay * self.baseline
            + (1 - self.baseline_decay) * reward
        )
        self._save()

    def get_stats(self) -> Dict:
        if not self.reward_history:
            return {"avg_reward": 0, "total_runs": 0, "baseline": self.baseline}
        recent = self.reward_history[-100:]
        return {
            "avg_reward": sum(r.reward for r in recent) / len(recent),
            "total_runs": len(self.reward_history),
            "baseline": self.baseline,
            "policy": self.policy.type_model_probs,
        }

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.data_path) or ".", exist_ok=True)
        data = {
            "baseline": self.baseline,
            "policy": self.policy.type_model_probs,
            "reward_count": len(self.reward_history),
        }
        with open(self.data_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path) as f:
                    data = json.load(f)
                self.baseline = data.get("baseline", 0.5)
                self.policy.type_model_probs = data.get("policy", {})
            except Exception:
                pass
