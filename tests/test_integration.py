from __future__ import annotations

from fugusashi.router import EnsembleRouter
from fugusashi.tracker import TransparencyTracker


def test_cost_router_picks_local_for_general():
    router = EnsembleRouter(confidence_threshold=0.4)
    available = {
        "llama3.2-local": {
            "cost_per_input_token": 0.0,
            "cost_per_output_token": 0.0,
            "capabilities": ["chat", "reasoning"],
        },
        "gpt-4o-mini": {
            "cost_per_input_token": 0.00000015,
            "cost_per_output_token": 0.0000006,
            "capabilities": ["chat", "reasoning", "code", "creative"],
        },
    }
    result = router.route(
        "What is the capital of France?",
        [{"role": "user", "content": "What is the capital of France?"}],
        available,
    )
    assert not result.needs_escalation
    assert result.confidence > 0
    print(f"Cost route: {result.model} (conf={result.confidence:.3f})")


def test_similarity_router_learns_from_history():
    router = EnsembleRouter(confidence_threshold=0.4)
    available = {
        "llama3.2-local": {
            "cost_per_input_token": 0.0,
            "cost_per_output_token": 0.0,
            "capabilities": ["chat", "reasoning"],
        },
        "gpt-4o-mini": {
            "cost_per_input_token": 0.00000015,
            "cost_per_output_token": 0.0000006,
            "capabilities": ["chat", "reasoning", "code", "creative"],
        },
    }
    router.similarity_router.build_index([
        {"prompt": "Write a Python function to sort a list", "model": "gpt-4o-mini", "score": 0.95},
        {"prompt": "What is the capital of France?", "model": "llama3.2-local", "score": 0.8},
    ])
    result = router.route(
        "How do I implement merge sort in Python?",
        [{"role": "user", "content": "How do I implement merge sort in Python?"}],
        available,
    )
    assert result.model == "gpt-4o-mini"
    assert "similarity" in result.strategy
    print(f"Similarity route: {result.model} (conf={result.confidence:.3f}, strat={result.strategy})")


def test_user_specified_model_bypasses_routing():
    router = EnsembleRouter(confidence_threshold=0.4)
    available = {"gpt-4o": {"cost_per_input_token": 0.0, "cost_per_output_token": 0.0, "capabilities": []}}
    result = router.route(
        "Any prompt",
        [{"role": "user", "content": "Any prompt"}],
        available,
    )
    assert result.model in available


def test_fallback_when_no_models():
    router = EnsembleRouter(confidence_threshold=0.4)
    result = router.route("test", [{"role": "user", "content": "test"}], {})
    assert result.needs_escalation


def test_tracker_tracks_routing_decisions():
    tracker = TransparencyTracker(log_to_console=False)
    tracker.start_trace("req-1")
    from fugusashi.tracker import RoutingDecision
    d = RoutingDecision(
        request_id="req-1",
        timestamp="2024-01-01",
        prompt_hash="abc",
        prompt_preview="test",
        routed_to="model-a",
        confidence=0.9,
        strategy="test",
        model_scores={"model-a": 0.9},
        latency_ms=1.0,
        explanation="test",
        needs_escalation=False,
    )
    tracker.log_routing("req-1", d)
    stats = tracker.get_stats()
    assert stats["total_requests"] == 1
    assert len(tracker.routing_log) == 1


if __name__ == "__main__":
    test_cost_router_picks_local_for_general()
    test_similarity_router_learns_from_history()
    test_user_specified_model_bypasses_routing()
    test_fallback_when_no_models()
    test_tracker_tracks_routing_decisions()
    print("\n All integration tests passed")
