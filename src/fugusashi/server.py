from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .api.routes import create_router
from .config import AppConfig
from .coordinator import CMAESRouter
from .federated import FederatedRouter, RoutingExplainer
from .feedback import FeedbackLoop
from .grpo import GRPOTrainer
from .orchestrator import MultiAgentOrchestrator
from .providers import ModelClient
from .router import EnsembleRouter
from .tracker import TransparencyTracker


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(
        title="Fugusashi",
        version="1.3.0",
        description="Intelligent model router — OpenAI-compatible API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    model_client = ModelClient([m.model_dump() for m in config.models])
    tracker = TransparencyTracker(log_to_console=config.observability.log_routing_decisions)

    router_engine = EnsembleRouter(
        embedding_model=config.tier1.router.embedding_model,
        confidence_threshold=config.tier1.router.confidence_threshold,
        fallback_model=config.default_model,
        prefer_local=config.tier1.router.prefer_local,
        model_dir=config.tier1.learned_router_model_dir,
        learned_router_enabled=config.tier1.learned_router_enabled,
        learned_confidence_threshold=config.tier1.learned_router_confidence_threshold,
    )

    feedback = FeedbackLoop()
    coordinator = CMAESRouter()
    federated = FederatedRouter()
    explainer = RoutingExplainer()

    orchestrator = None
    grpo = None
    if config.tier2.enabled:
        orchestrator = MultiAgentOrchestrator(
            model_client=model_client,
            planner_model=config.tier2.planner_model,
            synthesizer_model=config.tier2.synthesizer_model,
            max_subtasks=config.tier2.max_subtasks,
        )
        if config.tier2.grpo_enabled:
            grpo = GRPOTrainer(learning_rate=config.tier2.grpo_learning_rate)

    deps: Dict[str, Any] = {
        "config": config,
        "model_client": model_client,
        "tracker": tracker,
        "router": router_engine,
        "feedback": feedback,
        "coordinator": coordinator,
        "federated": federated,
        "explainer": explainer,
        "orchestrator": orchestrator,
        "grpo": grpo,
    }

    api_router = create_router(deps)
    app.include_router(api_router)

    dashboard_path = Path(__file__).parent / "static" / "dashboard.html"

    @app.get("/dashboard")
    async def dashboard():
        return FileResponse(dashboard_path)

    @app.on_event("startup")
    async def startup():
        pass

    return app
