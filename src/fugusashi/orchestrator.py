from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .providers import ModelClient


class TaskType(str, Enum):
    PLANNING = "planning"
    CODE = "code"
    REASONING = "reasoning"
    CREATIVE = "creative"
    FACTUAL = "factual"
    SYNTHESIS = "synthesis"


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    id: str
    description: str
    task_type: TaskType
    depends_on: List[str] = field(default_factory=list)
    assigned_model: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class TaskPlan:
    id: str
    original_prompt: str
    subtasks: List[SubTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    final_result: Optional[str] = None
    total_latency_ms: float = 0.0
    total_cost: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "original_prompt": self.original_prompt,
            "subtasks": [
                {
                    "id": st.id,
                    "description": st.description,
                    "task_type": st.task_type.value,
                    "depends_on": st.depends_on,
                    "assigned_model": st.assigned_model,
                    "status": st.status.value,
                    "result": st.result,
                    "error": st.error,
                    "latency_ms": st.latency_ms,
                }
                for st in self.subtasks
            ],
            "status": self.status.value,
            "final_result": self.final_result,
            "total_latency_ms": self.total_latency_ms,
        }


@dataclass
class OrchestratorResult:
    request_id: str
    plan: TaskPlan
    final_response: str
    total_latency_ms: float
    total_cost: float
    models_used: List[str]
    explanation: str


_PLANNER_SYSTEM = """You are a task planner. Given a user prompt, decompose it into 2-5 subtasks.
Each subtask should be assigned a type: code, reasoning, creative, factual, or synthesis.

Return a JSON array of subtasks. Each subtask:
{"description": "...", "task_type": "code|reasoning|creative|factual|synthesis", "depends_on": []}

- Subtasks with no dependencies can run in parallel
- "synthesis" tasks depend on all other tasks
- Keep descriptions self-contained (include all context from the original prompt)

Original prompt: {prompt}

Return ONLY the JSON array, no explanation."""

_TYPE_KEYWORDS = {
    TaskType.CODE: ["code", "function", "class", "implement", "write a", "python", "javascript", "debug", "refactor", "api", "script"],
    TaskType.REASONING: ["why", "how does", "explain", "analyze", "compare", "evaluate", "reason", "logic", "prove", "strategy"],
    TaskType.CREATIVE: ["write", "story", "poem", "creative", "brainstorm", "imagine", "design", "naming", "tagline"],
    TaskType.FACTUAL: ["what is", "who", "when", "where", "how many", "definition", "fact", "list", "name"],
}

_MODEL_CAPABILITIES: Dict[str, List[TaskType]] = {
    "code": [TaskType.CODE, TaskType.REASONING],
    "reasoning": [TaskType.REASONING, TaskType.FACTUAL],
    "creative": [TaskType.CREATIVE, TaskType.FACTUAL],
    "chat": [TaskType.FACTUAL, TaskType.CREATIVE, TaskType.REASONING],
}


def _classify_prompt(prompt: str) -> TaskType:
    lower = prompt.lower()
    for task_type, keywords in _TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return task_type
    return TaskType.REASONING


def _best_model_for_type(
    task_type: TaskType, models: Dict[str, dict]
) -> Optional[str]:
    scored: List[tuple[str, float]] = []
    for name, cfg in models.items():
        caps = cfg.get("capabilities", [])
        score = 0.0
        for cap in caps:
            if task_type in _MODEL_CAPABILITIES.get(cap, []):
                score += 1.0
        cost = cfg.get("cost_per_input_token", 0.0)
        if score > 0:
            scored.append((name, score - cost * 1e6))
    if scored:
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    return list(models.keys())[0] if models else None


class MultiAgentOrchestrator:
    """Tier 2: Decomposes prompts into subtasks, routes each to a specialist,
    and synthesizes the results."""

    def __init__(
        self,
        model_client: ModelClient,
        planner_model: Optional[str] = None,
        synthesizer_model: Optional[str] = None,
        max_subtasks: int = 5,
    ):
        self.model_client = model_client
        self.planner_model = planner_model
        self.synthesizer_model = synthesizer_model
        self.max_subtasks = max_subtasks
        self._history: List[OrchestratorResult] = []

    async def orchestrate(
        self, prompt: str, messages: Optional[List[Dict[str, str]]] = None
    ) -> OrchestratorResult:
        start = time.perf_counter()
        request_id = f"fugu-{uuid.uuid4().hex[:12]}"

        plan = await self._decompose(prompt, request_id)

        await self._assign_models(plan)

        await self._execute_subtasks(plan)

        final = await self._synthesize(plan, prompt)

        total_ms = (time.perf_counter() - start) * 1000
        plan.total_latency_ms = total_ms
        plan.status = TaskStatus.COMPLETED

        models_used = list(
            {st.assigned_model for st in plan.subtasks if st.assigned_model}
        )

        result = OrchestratorResult(
            request_id=request_id,
            plan=plan,
            final_response=final,
            total_latency_ms=total_ms,
            total_cost=self._calc_cost(plan),
            models_used=models_used,
            explanation=self._explain(plan),
        )
        self._history.append(result)
        return result

    async def _decompose(self, prompt: str, request_id: str) -> TaskPlan:
        plan = TaskPlan(id=request_id, original_prompt=prompt)

        if not self.planner_model:
            return self._rule_based_decompose(prompt, plan)

        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM.format(prompt=prompt)},
            {"role": "user", "content": prompt},
        ]

        try:
            resp, _, _, _, _ = await self.model_client.call_model(
                self.planner_model, messages, temperature=0.3, max_tokens=1024
            )
            content = resp.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            subtasks_raw = json.loads(content)
        except Exception:
            return self._rule_based_decompose(prompt, plan)

        for i, st in enumerate(subtasks_raw[: self.max_subtasks]):
            try:
                tt = TaskType(st.get("task_type", "reasoning"))
            except ValueError:
                tt = TaskType.REASONING
            plan.subtasks.append(
                SubTask(
                    id=f"sub-{i}",
                    description=st.get("description", prompt),
                    task_type=tt,
                    depends_on=st.get("depends_on", []),
                )
            )

        if not plan.subtasks:
            return self._rule_based_decompose(prompt, plan)

        return plan

    def _rule_based_decompose(self, prompt: str, plan: TaskPlan) -> TaskPlan:
        task_type = _classify_prompt(prompt)
        plan.subtasks.append(
            SubTask(
                id="sub-0",
                description=prompt,
                task_type=task_type,
                depends_on=[],
            )
        )
        return plan

    async def _assign_models(self, plan: TaskPlan) -> None:
        models = self.model_client.get_available_models()
        for st in plan.subtasks:
            st.assigned_model = _best_model_for_type(st.task_type, models)
            st.status = TaskStatus.ASSIGNED

    async def _execute_subtasks(self, plan: TaskPlan) -> None:
        completed: Dict[str, SubTask] = {}
        remaining = list(plan.subtasks)

        while remaining:
            ready = [
                st
                for st in remaining
                if all(d in completed for d in st.depends_on)
            ]
            if not ready:
                for st in remaining:
                    st.status = TaskStatus.FAILED
                    st.error = "Circular dependency or missing dependency"
                break

            import asyncio

            async def _run_task(st: SubTask) -> None:
                st.status = TaskStatus.RUNNING
                ctx = self._build_context(st, completed)
                messages = [
                    {"role": "system", "content": ctx},
                    {"role": "user", "content": st.description},
                ]
                try:
                    resp, ms, pt, ct, _ = await self.model_client.call_model(
                        st.assigned_model, messages, temperature=0.5
                    )
                    st.result = resp.choices[0].message.content
                    st.latency_ms = ms
                    st.prompt_tokens = pt
                    st.completion_tokens = ct
                    st.status = TaskStatus.COMPLETED
                except Exception as e:
                    st.error = str(e)
                    st.status = TaskStatus.FAILED

            await asyncio.gather(*[_run_task(st) for st in ready])

            for st in ready:
                completed[st.id] = st
                remaining.remove(st)

    def _build_context(
        self, st: SubTask, completed: Dict[str, SubTask]
    ) -> str:
        parts = [
            "You are a specialist agent. Complete the assigned subtask accurately.",
            f"Subtask type: {st.task_type.value}",
        ]
        if st.depends_on:
            deps = []
            for dep_id in st.depends_on:
                dep = completed.get(dep_id)
                if dep and dep.result:
                    deps.append(f"Result from '{dep.description[:80]}': {dep.result[:500]}")
            if deps:
                parts.append("Context from previous subtasks:")
                parts.extend(deps)
        return "\n".join(parts)

    async def _synthesize(self, plan: TaskPlan, prompt: str) -> str:
        results = [
            f"[{st.task_type.value}] {st.description[:80]}:\n{st.result or st.error or 'No result'}"
            for st in plan.subtasks
            if st.result or st.error
        ]

        if len(results) == 1:
            return plan.subtasks[0].result or plan.subtasks[0].error or "No result"

        synth_model = self.synthesizer_model or self._pick_synthesizer()
        context = "\n\n".join(results)
        synth_prompt = (
            f"Original request: {prompt}\n\n"
            f"Specialist results:\n{context}\n\n"
            "Synthesize a single coherent response that addresses the original request "
            "using the specialist results above. Be concise and accurate."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a synthesis agent. Combine specialist outputs into one coherent response.",
            },
            {"role": "user", "content": synth_prompt},
        ]

        try:
            resp, _, _, _, _ = await self.model_client.call_model(
                synth_model, messages, temperature=0.3, max_tokens=2048
            )
            return resp.choices[0].message.content
        except Exception:
            parts = []
            for st in plan.subtasks:
                if st.result:
                    parts.append(st.result)
            return "\n\n".join(parts) if parts else "All subtasks failed."

    def _pick_synthesizer(self) -> str:
        models = self.model_client.get_available_models()
        for name, cfg in models.items():
            caps = cfg.get("capabilities", [])
            if "reasoning" in caps or "chat" in caps:
                return name
        return list(models.keys())[0] if models else "gpt-4o-mini"

    def _calc_cost(self, plan: TaskPlan) -> float:
        total = 0.0
        for st in plan.subtasks:
            if st.assigned_model:
                cfg = self.model_client.model_configs.get(st.assigned_model, {})
                total += st.prompt_tokens * cfg.get("cost_per_input_token", 0.0)
                total += st.completion_tokens * cfg.get("cost_per_output_token", 0.0)
        return total

    def _explain(self, plan: TaskPlan) -> str:
        lines = [f"Decomposed into {len(plan.subtasks)} subtask(s):"]
        for st in plan.subtasks:
            status = "ok" if st.status == TaskStatus.COMPLETED else "failed"
            lines.append(
                f"  [{st.task_type.value}] {st.assigned_model} "
                f"({st.latency_ms:.0f}ms) - {status}"
            )
        return "\n".join(lines)

    def get_history(self) -> List[OrchestratorResult]:
        return list(self._history)
