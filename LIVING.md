# LIVING.md — Fugusashi Development Walkthrough

> This document is the complete story of how Fugusashi is built, why each decision was made, and where it's going. It is updated with every significant change. Anyone can read this to understand the project from any point in time.

---

## Table of Contents

1. [The Vision](#the-vision)
2. [Architecture Decisions](#architecture-decisions)
3. [Phase 1: The Router (Week 1)](#phase-1-the-router)
4. [Phase 2: Transparency & Observability (Week 1-2)](#phase-2-transparency--observability)
5. [Phase 3: Feedback Loop (Week 2)](#phase-3-feedback-loop)
6. [Phase 4: Benchmarking (Week 2)](#phase-4-benchmarking)
7. [Phase 5: Dashboard (Week 2)](#phase-5-dashboard)
8. [What's Next](#whats-next)
9. [Changelog](#changelog)

---

## The Vision

Fugu by Sakana AI is a two-mode system:
- **Fast mode**: A lightweight router picks the best single model for each prompt
- **Ultra mode**: A multi-agent orchestrator decomposes hard tasks and assigns them to specialist models

The problem: Fugu is a black box. You can't see why it routes to which model, you can't train it on your own data, you can't run it locally, and you pay $5-30 per million tokens.

**Fugusashi** (Japanese for "not bound / unrestrained") is the open alternative:
- Every routing decision is transparent
- Runs entirely on-premise with local models
- Learns from your traffic via a feedback loop
- Community-owned preference data
- Free. MIT licensed. Yours.

---

## Architecture Decisions

### Why two tiers?

Fugu's two modes map cleanly to two tiers:
- **Tier 1 (Router)**: Fast, single-model routing. Handles 90% of requests.
- **Tier 2 (Orchestrator)**: Slow, multi-agent planning. Only for complex tasks that need decomposition.

This separation keeps the common case fast while enabling complex workflows when needed.

### Why LiteLLM?

We don't want to write provider integrations. LiteLLM gives us 100+ providers (OpenAI, Anthropic, Ollama, etc.) with a unified API. We use it as the translation layer — our router picks a model, LiteLLM handles the actual call.

### Why sentence-transformers for similarity?

The similarity router needs to compare prompts to historical examples. Sentence-transformers gives us good embeddings fast on CPU. Alternatives considered:
- **OpenAI embeddings**: Requires API key, costs money
- **TF-IDF**: Doesn't capture semantics well
- **Fine-trained classifier**: Needs labeled data upfront

Sentence-transformers hits the sweet spot: free, fast, good enough.

### Why an ensemble with priority chain?

No single routing strategy is always right:
- Similarity is smart but needs training data
- Cost-based is always available but dumb
- Fallback ensures something always works

The priority chain (similarity → cost → fallback) means we use the smartest strategy that has enough confidence, and fall back gracefully.

---

## Phase 1: The Router

### What we built

The core routing engine with three strategies:

**`CostRouter`** — Routes based on capability matching + cost:
```
prompt → keyword analysis → capability need → score each model → pick best
```

- Detects code, math, creative, factual, general prompts via keyword matching
- Scores models by: capability match (0.5-0.9) minus cost fraction
- Respects `prefer_local` for air-gapped deployments
- Always available (no training data needed)

**`SimilarityRouter`** — Routes based on historical prompts:
```
prompt → embed → find K nearest neighbors → vote by model quality → pick best
```

- Uses `all-MiniLM-L6-v2` embeddings (fast, good quality)
- Only counts votes above similarity threshold (0.2) to avoid noise
- Uses max score (not mean) so strong matches aren't diluted
- Falls back to CostRouter when no good match exists

**`EnsembleRouter`** — Priority chain:
```
try similarity (if confidence ≥ threshold) → try cost → fallback to default
```

- First strategy with sufficient confidence wins
- Guaranteed to always return a result
- Tracks which strategy made each decision

### Key files
- `src/fugusashi/router/interface.py` — Abstract `BaseRouter` protocol
- `src/fugusashi/router/strategies.py` — `CostRouter`, `SimilarityRouter`, `FallbackRouter`
- `src/fugusashi/router/ensemble.py` — `EnsembleRouter`

### Design decisions

1. **Threshold = 0.4**: Below this, the router isn't confident enough and escalates. This is configurable.
2. **Max score, not mean**: When 5 neighbors vote but only 1 is relevant, the mean dilutes the signal. Max preserves strong matches.
3. **Min similarity = 0.2**: Neighbors below this similarity are noise. Ignored.

---

## Phase 2: Transparency & Observability

### What we built

**`TransparencyTracker`** — Records every routing decision and model call:
- `RoutingDecision`: which model, confidence, strategy, scores, explanation
- `ModelCallRecord`: tokens, cost, latency, status
- `RequestTrace`: full lifecycle of a request

**API endpoints**:
- `GET /v1/routing/decisions` — Recent routing decisions
- `GET /v1/stats` — Aggregated stats (cost, tokens, per-model)
- `GET /v1/trace/{request_id}` — Full trace for debugging

**Response embedding**: Every `/v1/chat/completions` response includes a `routing_decision` field so clients can see why a particular model was chosen.

### Why this matters

Fugu is a black box. You send a prompt, get a response, and have no idea which model handled it or why. Fugusashi exposes everything:
- Which model was picked and why
- Confidence score
- Alternative models considered
- Routing latency
- Token usage and cost

This enables debugging, cost optimization, and trust.

---

## Phase 3: Feedback Loop

### The missing piece

Fugu's router is static — it doesn't learn from outcomes. If it routes a coding prompt to a model that produces bad code, it won't remember that next time.

Fugusashi's feedback loop closes this gap:

```
Route → Execute → Evaluate → Learn
```

### What we built

**`FeedbackLoop`** in `src/fugusashi/feedback.py`:
- Records every outcome (success/failure, cost, latency, tokens)
- Accepts user ratings (1-5) via `POST /v1/feedback/rate`
- Builds similarity index from accumulated outcomes
- `POST /v1/feedback/retrain` — Rebuild the similarity router from feedback
- `GET /v1/feedback/stats` — Outcome statistics, model rankings
- `GET /v1/feedback/rankings` — Per-model win rates, costs, ratings

### How learning works

1. Every request is recorded with its outcome
2. User ratings (optional) override automatic scoring
3. When retrained, the similarity router rebuilds its index from all outcomes
4. Prompts that got high ratings reinforce their model choice
5. Prompts that failed are downweighted

### Data storage

Outcomes are stored in `.fugusashi_data/outcomes.jsonl` — append-only, easy to inspect, easy to share. The similarity index is rebuilt on demand.

### Key files
- `src/fugusashi/feedback.py` — `FeedbackLoop`, `OutcomeRecord`, `ModelScore`
- `src/fugusashi/api/routes.py` — `/v1/feedback/*` endpoints

---

## Phase 4: Benchmarking

### What we built

`fugusashi benchmark` — A CLI tool that measures routing quality:

```bash
fugusashi benchmark                      # Default 20-sample dataset
fugusashi benchmark --train              # With seeded training data
fugusashi benchmark --train --verbose    # Per-sample results
fugusashi benchmark --json               # Machine-readable output
fugusashi benchmark -d my_data.jsonl     # Custom dataset
```

### Metrics

- **Accuracy**: % of prompts routed to the expected model
- **Cost vs ideal**: How much you save compared to always using the best model
- **Avg routing latency**: Time to make the routing decision
- **Strategy distribution**: How often each strategy is used
- **Per-category breakdown**: Accuracy by task type (code, creative, factual, etc.)

### Default dataset

20 prompts across categories (code, factual, creative, explanation, general) with expected model labels. Designed to show the difference between cost-based routing and similarity-based routing.

### Results

| Metric | Without Training | With Training |
|---|---|---|
| Accuracy | 70% | **85%** |
| Code accuracy | 60% | **90%** |
| Strategy | 100% cost | 60% cost / 40% similarity |
| Routing latency | <1ms | ~18ms |

Training data improved accuracy by 15 points. The similarity router correctly routes code tasks to capable models while keeping casual chat on free local models.

### Custom datasets

JSONL format:
```jsonl
{"prompt":"How do I center a div?","expected_model":"gpt-4o-mini","category":"code"}
{"prompt":"What is 2+2?","expected_model":"llama3.2-local","category":"factual"}
```

---

## Phase 5: Dashboard

### What we built

A live web dashboard at `GET /dashboard`:

- **Overview**: total requests, tokens, cost, avg routing latency
- **Model Usage**: bar chart of which models are being picked
- **Strategy Distribution**: cost vs similarity routing breakdown
- **Recent Decisions**: live table with confidence, strategy, model

Auto-refreshes every 3 seconds. Pure HTML/CSS/JS, no build step, no dependencies.

### Key files
- `src/fugusashi/static/dashboard.html` — Single-file dashboard
- `src/fugusashi/server.py` — Serves dashboard at `/dashboard`

---

## What's Next

### Tier 2: Multi-Agent Orchestrator

The big one. A planning model that:
1. Takes a complex task
2. Decomposes it into subtasks
3. Assigns each subtask to the best specialist model
4. Synthesizes results
5. Learns from outcomes (GRPO-style reinforcement learning)

This is what makes Fugu Ultra powerful. Our version will be open and transparent — you'll see the plan, the assignments, and the synthesis.

### Community Preference Sharing

Export/import routing datasets. Share what works for your use case. Build a community-curated dataset that makes the router better for everyone.

### Plugin System

Custom routing strategies. The interface is already defined (`BaseRouter`). Next step is a plugin registry that auto-discovers strategies.

---

## Changelog

### v0.2.0 — Feedback Loop

- Added `FeedbackLoop` class that records every routing outcome
- Added `/v1/feedback/rate` — User ratings (1-5)
- Added `/v1/feedback/retrain` — Rebuild similarity index from feedback
- Added `/v1/feedback/stats` — Outcome statistics
- Added `/v1/feedback/rankings` — Per-model win rates
- Outcomes persisted to `.fugusashi_data/outcomes.jsonl`
- Feedback recording integrated into chat completion flow
- Fixed SimilarityRouter scoring: use max instead of mean, filter by min similarity
- Accuracy improved: 70% → 85% with training data

### v0.1.0 — Initial Release

- EnsembleRouter with CostRouter, SimilarityRouter, FallbackRouter
- OpenAI-compatible API (`/v1/chat/completions`)
- Transparency endpoints (`/v1/routing/decisions`, `/v1/stats`, `/v1/trace`)
- Training data ingestion (`/v1/routing/training`)
- Model fallback chain
- LiteLLM integration for 100+ providers
- Web dashboard at `/dashboard`
- Benchmark CLI tool
- Integration tests (5 passing)
- MIT licensed

---

## How to Contribute

1. Fork the repo
2. Add a routing strategy by implementing `BaseRouter`
3. Add benchmarks for your use case
4. Share your training data

---

*Last updated: 2026-06-23*
