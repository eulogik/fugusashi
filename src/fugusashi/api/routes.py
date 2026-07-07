from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..tracker import RoutingDecision
from ..coordinator import Task


class ChatMessage(BaseModel):
    role: str
    content: str


class TrainingExample(BaseModel):
    prompt: str
    model: str
    score: float = 1.0


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    user: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]
    routing_decision: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "fugusashi"
    description: str = ""
    capabilities: List[str] = []
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0


def create_router(deps) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    @router.get("/v1/models")
    async def list_models():
        available = deps["model_client"].get_available_models()
        models = []
        for name, cfg in available.items():
            models.append(ModelInfo(
                id=name,
                created=int(datetime.utcnow().timestamp()),
                description=cfg.get("description", ""),
                capabilities=cfg.get("capabilities", []),
                cost_per_input_token=cfg.get("cost_per_input_token", 0.0),
                cost_per_output_token=cfg.get("cost_per_output_token", 0.0),
            ))
        return {"object": "list", "data": models}

    @router.post("/v1/chat/completions")
    async def chat_completion(body: ChatCompletionRequest, raw_request: Request):
        request_id = f"fugu-{uuid.uuid4().hex[:12]}"
        tracker = deps["tracker"]
        model_client = deps["model_client"]
        router_engine = deps["router"]
        config = deps["config"]

        tracker.start_trace(request_id)
        prompt = body.messages[-1].content if body.messages else ""
        prompt_preview = prompt[:200]

        if body.model == "coordinator":
            coordinator = deps.get("coordinator")
            if coordinator:
                coord_result = coordinator.route(prompt)
                selected_model = coord_result.model
                routing_result = RoutingDecision(
                    request_id=request_id,
                    timestamp=datetime.utcnow().isoformat(),
                    prompt_hash=str(hash(prompt)),
                    prompt_preview=prompt_preview,
                    routed_to=selected_model,
                    confidence=coord_result.confidence,
                    strategy=coord_result.strategy,
                    model_scores=coord_result.scores,
                    latency_ms=coord_result.latency_ms,
                    explanation=f"CMA-ES coordinator routing (conf={coord_result.confidence:.2f})",
                    needs_escalation=coord_result.confidence < 0.3,
                )
            else:
                selected_model = config.default_model
                routing_result = RoutingDecision(
                    request_id=request_id,
                    timestamp=datetime.utcnow().isoformat(),
                    prompt_hash=str(hash(prompt)),
                    prompt_preview=prompt_preview,
                    routed_to=selected_model,
                    confidence=0.5,
                    strategy="fallback",
                    model_scores={},
                    latency_ms=0.0,
                    explanation="Coordinator not available, using default",
                    needs_escalation=False,
                )
        elif body.model and body.model != "auto":
            selected_model = body.model
            routing_result = RoutingDecision(
                request_id=request_id,
                timestamp=datetime.utcnow().isoformat(),
                prompt_hash=str(hash(prompt)),
                prompt_preview=prompt_preview,
                routed_to=selected_model,
                confidence=1.0,
                strategy="user-specified",
                model_scores={selected_model: 1.0},
                latency_ms=0.0,
                explanation="User explicitly specified the model",
                needs_escalation=False,
            )
        else:
            available = model_client.get_available_models()
            threshold = config.tier1.router.confidence_threshold
            result = router_engine.route(
                prompt=prompt,
                messages=[m.model_dump() for m in body.messages],
                available_models=available,
                threshold=threshold,
            )

            if result.needs_escalation and config.tier2.enabled and config.tier2.auto_escalate:
                orchestrator = deps.get("orchestrator")
                if orchestrator:
                    orch_result = await orchestrator.orchestrate(prompt)
                    return ChatCompletionResponse(
                        request_id if False else f"fugu-{uuid.uuid4().hex[:12]}",
                        created=int(datetime.utcnow().timestamp()),
                        model="orchestrator",
                        choices=[{"index": 0, "message": {"role": "assistant", "content": orch_result.final_response}, "finish_reason": "stop"}],
                        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                        routing_decision={
                            "model": "orchestrator",
                            "confidence": 1.0,
                            "strategy": "tier2-orchestration",
                            "latency_ms": orch_result.total_latency_ms,
                            "explanation": orch_result.explanation,
                            "plan": orch_result.plan.to_dict(),
                        },
                    )

            selected_model = result.model
            routing_result = RoutingDecision(
                request_id=request_id,
                timestamp=datetime.utcnow().isoformat(),
                prompt_hash=str(hash(prompt)),
                prompt_preview=prompt_preview,
                routed_to=selected_model,
                confidence=result.confidence,
                strategy=result.strategy,
                model_scores=result.scores,
                latency_ms=result.latency_ms,
                explanation=result.explanation,
                needs_escalation=result.needs_escalation,
            )

        tracker.log_routing(request_id, routing_result)

        if body.stream:
            from fastapi.responses import StreamingResponse

            async def stream_generator():
                full_content = ""
                prompt_tokens = 0
                completion_tokens = 0

                try:
                    async for chunk in model_client.call_model_stream(
                        model_name=selected_model,
                        messages=[m.model_dump() for m in body.messages],
                        temperature=body.temperature,
                        max_tokens=body.max_tokens,
                    ):
                        if hasattr(chunk, "choices") and chunk.choices:
                            delta = chunk.choices[0].delta
                            if hasattr(delta, "content") and delta.content:
                                full_content += delta.content
                        chunk_data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                    yield f"data: {json.dumps({'routing_decision': {
                        'model': selected_model,
                        'confidence': routing_result.confidence,
                        'strategy': routing_result.strategy,
                        'latency_ms': routing_result.latency_ms,
                        'explanation': routing_result.explanation,
                    }})}\n\n"

                    yield "data: [DONE]\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

                tracker.log_model_call(
                    request_id=request_id,
                    model=selected_model,
                    provider="",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    status="success",
                )
                tracker.finish_trace(request_id)

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        models_to_try = [selected_model]
        if selected_model != config.default_model:
            models_to_try.append(config.default_model)

        last_error = None
        response = None
        for fallback_idx, model_to_try in enumerate(models_to_try):
            try:
                response, latency, prompt_tokens, completion_tokens, provider = (
                    await model_client.call_model(
                        model_name=model_to_try,
                        messages=[m.model_dump() for m in body.messages],
                        temperature=body.temperature,
                        max_tokens=body.max_tokens,
                    )
                )

                tracker.log_model_call(
                    request_id=request_id,
                    model=model_to_try,
                    provider=provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost=0.0,
                    latency_ms=latency,
                    status="success" if fallback_idx == 0 else "fallback_success",
                )

                feedback = deps.get("feedback")
                if feedback:
                    feedback.record_routing(
                        prompt=prompt,
                        routed_to=model_to_try,
                        confidence=routing_result.confidence,
                        strategy=routing_result.strategy,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost=0.0,
                        latency_ms=latency,
                        error=False,
                        auto_retrain=True,
                        retrain_interval=10,
                        router=router_engine,
                    )

                tracker.finish_trace(request_id)
                break
            except Exception as e:
                last_error = e
                tracker.log_model_call(
                    request_id=request_id,
                    model=model_to_try,
                    provider="",
                    status="error" if fallback_idx == 0 else "fallback_error",
                    error=str(e),
                )
                feedback = deps.get("feedback")
                if feedback:
                    feedback.record_routing(
                        prompt=prompt,
                        routed_to=model_to_try,
                        confidence=routing_result.confidence,
                        strategy=routing_result.strategy,
                        error=True,
                        auto_retrain=True,
                        retrain_interval=10,
                        router=router_engine,
                    )
                continue

        if response is None:
            raise HTTPException(status_code=502, detail=f"All models failed. Last error: {last_error}")

        response_dict = response.model_dump() if hasattr(response, "model_dump") else response

        raw_usage = response_dict.get("usage", {})
        if raw_usage is None:
            raw_usage = {}
        sanitized_usage = {
            k: (v if v is not None else 0)
            for k, v in raw_usage.items()
        }

        return ChatCompletionResponse(
            id=request_id,
            created=int(datetime.utcnow().timestamp()),
            model=model_to_try,
            choices=response_dict.get("choices", []),
            usage=sanitized_usage,
            routing_decision={
                    "model": selected_model,
                    "confidence": routing_result.confidence,
                    "strategy": routing_result.strategy,
                    "latency_ms": routing_result.latency_ms,
                    "explanation": routing_result.explanation,
                },
            )

    @router.get("/v1/routing/decisions")
    async def get_routing_decisions(limit: int = 20):
        decisions = deps["tracker"].routing_log[-limit:]
        return {
            "object": "list",
            "data": [
                {
                    "request_id": d.request_id,
                    "timestamp": d.timestamp,
                    "prompt_preview": d.prompt_preview,
                    "routed_to": d.routed_to,
                    "confidence": d.confidence,
                    "strategy": d.strategy,
                    "latency_ms": d.latency_ms,
                    "explanation": d.explanation,
                }
                for d in decisions
            ],
        }

    @router.get("/v1/stats")
    async def get_stats():
        return deps["tracker"].get_stats()

    @router.get("/v1/trace/{request_id}")
    async def get_trace(request_id: str):
        trace = deps["tracker"].get_trace(request_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace

    @router.post("/v1/routing/training")
    async def add_training_data(examples: List[TrainingExample]):
        router_engine = deps["router"]
        history = [
            {"prompt": ex.prompt, "model": ex.model, "score": ex.score}
            for ex in examples
        ]
        router_engine.similarity_router.build_index(history)
        return {"status": "ok", "indexed": len(history)}

    @router.post("/v1/feedback/rate")
    async def rate_outcome(request: Request):
        body = await request.json()
        request_id = body.get("request_id", "")
        rating = int(body.get("rating", 3))
        feedback: Any = deps.get("feedback")
        if not feedback:
            return {"status": "error", "message": "feedback not enabled"}
        trace = feedback.outcomes
        for outcome in reversed(trace):
            if outcome.timestamp and outcome.timestamp.endswith(request_id[-6:]):
                feedback.record_user_rating(outcome, rating)
                return {"status": "ok", "rating": rating}
        return {"status": "not_found", "request_id": request_id}

    @router.post("/v1/feedback/retrain")
    async def retrain():
        feedback: Any = deps.get("feedback")
        router_engine = deps["router"]
        if not feedback:
            return {"status": "error", "message": "feedback not enabled"}
        feedback.build_similarity_index(router_engine.similarity_router)
        data = feedback.get_retraining_data()
        return {"status": "ok", "retrained_on": len(data)}

    @router.get("/v1/feedback/stats")
    async def feedback_stats():
        feedback: Any = deps.get("feedback")
        if not feedback:
            return {"status": "error", "message": "feedback not enabled"}
        return feedback.get_stats()

    @router.get("/v1/feedback/rankings")
    async def model_rankings():
        feedback: Any = deps.get("feedback")
        if not feedback:
            return {"status": "error", "message": "feedback not enabled"}
        return feedback.get_model_rankings()

    @router.post("/v1/coordinator/evolve")
    async def evolve_coordinator(request: Request):
        body = await request.json()
        tasks_data = body.get("tasks", [])
        fast = body.get("fast", True)
        coordinator = deps.get("coordinator")
        if not coordinator:
            return {"status": "error", "message": "coordinator not enabled"}
        tasks = [Task(p=t["prompt"], category=t.get("category", "general")) for t in tasks_data]
        coordinator.evolve(tasks, fast=fast)
        return {"status": "ok", "stats": coordinator.get_stats()}

    @router.get("/v1/coordinator/stats")
    async def coordinator_stats():
        coordinator = deps.get("coordinator")
        if not coordinator:
            return {"status": "error", "message": "coordinator not enabled"}
        return coordinator.get_stats()

    @router.post("/v1/coordinator/route")
    async def coordinator_route(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        coordinator = deps.get("coordinator")
        if not coordinator:
            return {"status": "error", "message": "coordinator not enabled"}
        result = coordinator.route(prompt)
        return {
            "model": result.model,
            "confidence": result.confidence,
            "scores": result.scores,
            "strategy": result.strategy,
            "latency_ms": result.latency_ms,
        }

    @router.post("/v1/federated/register")
    async def federated_register(request: Request):
        body = await request.json()
        client_id = body.get("client_id", "")
        metadata = body.get("metadata", {})
        federated = deps.get("federated")
        if not federated:
            return {"status": "error", "message": "federated not enabled"}
        federated.register_client(client_id, metadata)
        return {"status": "ok", "client_id": client_id}

    @router.post("/v1/federated/submit")
    async def federated_submit(request: Request):
        body = await request.json()
        client_id = body.get("client_id", "")
        weights = body.get("weights", [])
        n_samples = body.get("n_samples", 0)
        metrics = body.get("metrics", {})
        federated = deps.get("federated")
        if not federated:
            return {"status": "error", "message": "federated not enabled"}
        result = federated.submit_update(
            client_id,
            np.array(weights),
            n_samples,
            metrics,
        )
        return result

    @router.post("/v1/federated/aggregate")
    async def federated_aggregate():
        federated = deps.get("federated")
        if not federated:
            return {"status": "error", "message": "federated not enabled"}
        result = federated.aggregate()
        if result is None:
            return {"status": "waiting", "message": "Not enough updates to aggregate"}
        return {
            "status": "ok",
            "round": result.round_id,
            "participants": result.participating_clients,
        }

    @router.get("/v1/federated/stats")
    async def federated_stats():
        federated = deps.get("federated")
        if not federated:
            return {"status": "error", "message": "federated not enabled"}
        return federated.get_client_stats()

    @router.post("/v1/explain")
    async def explain_routing(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        coordinator = deps.get("coordinator")
        explainer = deps.get("explainer")
        if not coordinator or not explainer:
            return {"status": "error", "message": "explanation not enabled"}
        decision = coordinator.route(prompt)
        explanation = explainer.explain(prompt, decision, decision.scores)
        return {
            "model": decision.model,
            "confidence": decision.confidence,
            "scores": decision.scores,
            "explanation": explanation,
        }

    @router.post("/v1/orchestrate")
    async def orchestrate(request: Request):
        body = await request.json()
        prompt = body.get("prompt", "")
        messages = body.get("messages", [])
        if not prompt and messages:
            prompt = messages[-1].get("content", "")

        orchestrator = deps.get("orchestrator")
        if not orchestrator:
            return {"status": "error", "message": "Tier 2 orchestrator not enabled. Set tier2.enabled=true in config."}

        result = await orchestrator.orchestrate(prompt, messages or None)
        return {
            "request_id": result.request_id,
            "response": result.final_response,
            "plan": result.plan.to_dict(),
            "total_latency_ms": result.total_latency_ms,
            "total_cost": result.total_cost,
            "models_used": result.models_used,
            "explanation": result.explanation,
        }

    @router.get("/v1/orchestration/trace/{plan_id}")
    async def orchestration_trace(plan_id: str):
        orchestrator = deps.get("orchestrator")
        if not orchestrator:
            return {"status": "error", "message": "Tier 2 not enabled"}
        for result in orchestrator.get_history():
            if result.request_id == plan_id:
                return result.plan.to_dict()
        raise HTTPException(status_code=404, detail="Plan not found")

    @router.get("/v1/orchestration/history")
    async def orchestration_history(limit: int = 10):
        orchestrator = deps.get("orchestrator")
        if not orchestrator:
            return {"status": "error", "message": "Tier 2 not enabled"}
        history = orchestrator.get_history()[-limit:]
        return {
            "total": len(orchestrator.get_history()),
            "plans": [
                {
                    "id": r.request_id,
                    "subtasks": len(r.plan.subtasks),
                    "models_used": r.models_used,
                    "latency_ms": r.total_latency_ms,
                    "cost": r.total_cost,
                }
                for r in history
            ],
        }

    @router.get("/v1/orchestration/grpo/stats")
    async def grpo_stats():
        grpo = deps.get("grpo")
        if not grpo:
            return {"status": "error", "message": "GRPO not enabled"}
        return grpo.get_stats()

    @router.post("/v1/orchestration/grpo/score")
    async def grpo_score(request: Request):
        body = await request.json()
        plan_id = body.get("plan_id", "")
        grpo = deps.get("grpo")
        orchestrator = deps.get("orchestrator")
        if not grpo or not orchestrator:
            return {"status": "error", "message": "GRPO or orchestrator not enabled"}
        for result in orchestrator.get_history():
            if result.request_id == plan_id:
                reward = grpo.score(result)
                grpo.update_policy(result, reward.reward)
                return {
                    "plan_id": plan_id,
                    "reward": reward.reward,
                    "decomposition": reward.decomposition_score,
                    "routing": reward.routing_score,
                    "synthesis": reward.synthesis_score,
                    "baseline": grpo.baseline,
                }
        raise HTTPException(status_code=404, detail="Plan not found")

    return router
