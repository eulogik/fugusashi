from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RoutingDecision:
    request_id: str
    timestamp: str
    prompt_hash: str
    prompt_preview: str
    routed_to: str
    confidence: float
    strategy: str
    model_scores: Dict[str, float]
    latency_ms: float
    explanation: str
    needs_escalation: bool


@dataclass
class ModelCallRecord:
    request_id: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    status: str = "pending"
    error: Optional[str] = None


@dataclass
class RequestTrace:
    request_id: str
    start_time: str
    end_time: Optional[str] = None
    total_latency_ms: float = 0.0
    total_cost: float = 0.0
    routing: Optional[RoutingDecision] = None
    model_calls: List[ModelCallRecord] = field(default_factory=list)


class TransparencyTracker:
    def __init__(self, log_to_console: bool = True):
        self.traces: Dict[str, RequestTrace] = {}
        self.routing_log: List[RoutingDecision] = []
        self._log_to_console = log_to_console
        self._stats = defaultdict(lambda: {"calls": 0, "total_cost": 0.0, "total_tokens": 0})

    def start_trace(self, request_id: str) -> RequestTrace:
        trace = RequestTrace(
            request_id=request_id,
            start_time=datetime.utcnow().isoformat(),
        )
        self.traces[request_id] = trace
        return trace

    def log_routing(self, request_id: str, decision: RoutingDecision):
        trace = self.traces.get(request_id)
        if trace:
            trace.routing = decision
        self.routing_log.append(decision)
        if self._log_to_console:
            print(
                f"[ROUTER] {decision.routed_to} "
                f"(conf={decision.confidence:.2f}, "
                f"strat={decision.strategy}, "
                f"lat={decision.latency_ms:.1f}ms)"
            )

    def log_model_call(
        self,
        request_id: str,
        model: str,
        provider: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
        status: str = "success",
        error: Optional[str] = None,
    ):
        trace = self.traces.get(request_id)
        record = ModelCallRecord(
            request_id=request_id,
            model=model,
            provider=provider,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
            status=status,
            error=error,
        )
        if trace:
            trace.model_calls.append(record)
            trace.total_cost += cost
            trace.total_latency_ms += latency_ms

        s = self._stats[model]
        s["calls"] += 1
        s["total_cost"] += cost
        s["total_tokens"] += prompt_tokens + completion_tokens

        if self._log_to_console:
            status_icon = "✓" if status == "success" else "✗"
            print(f"[MODEL] {status_icon} {model} {prompt_tokens}→{completion_tokens}tok ${cost:.6f} {latency_ms:.0f}ms")

    def finish_trace(self, request_id: str):
        trace = self.traces.get(request_id)
        if trace:
            trace.end_time = datetime.utcnow().isoformat()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": len(self.traces),
            "total_cost": sum(t.total_cost for t in self.traces.values()),
            "total_tokens": sum(
                sum(mc.total_tokens for mc in t.model_calls) for t in self.traces.values()
            ),
            "per_model": dict(self._stats),
            "recent_routes": [
                asdict(d) for d in self.routing_log[-50:]
            ],
        }

    def get_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        trace = self.traces.get(request_id)
        if trace:
            return asdict(trace)
        return None
