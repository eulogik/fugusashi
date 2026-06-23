<div align="center">

# Fugusashi

**Intelligent Model Router & Orchestrator**

*Open-source alternative to Fugu — transparent, self-hosting, learning.*

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-living-blue.svg)](LIVING.md)

[Architecture](#architecture) •
[Quickstart](#quickstart) •
[API](#api) •
[Dashboard](#dashboard) •
[Benchmarking](#benchmarking) •
[Feedback Loop](#feedback-loop) •
[Development Log](LIVING.md)

</div>

---

## Why Fugusashi?

| Feature | Fugu | Fugusashi |
|---|---|---|
| Transparency | Black box | Every routing decision visible |
| Cost | $5-30/M tokens | Self-hosted, pay only for model APIs |
| Model Pool | Fixed | You control which models to include |
| Training Data | Proprietary | Community-curated + your own traffic |
| Deployment | Cloud-only | Local-first, air-gapped capable |
| Customization | None | Fine-tune router on your own traffic |
| Feedback Loop | None | **Learns from every request** |
| Dashboard | Proprietary | Open web dashboard |

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Your Application (OpenAI-compatible)    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│  TIER 1: ROUTER (CPU, <20ms)            │
│  - SimilarityRouter (learns over time)  │
│  - CostRouter (capability + price)      │
│  - EnsembleRouter (priority chain)      │
│  - Routes to single model OR            │
│    escalates to Tier 2                  │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────┐              ┌──────────────┐
│ Single  │              │ TIER 2:      │
│ Model   │              │ ORCHESTRATOR │
│ Call    │              │ (Phase 2)    │
└─────────┘              └──────────────┘
```

### Two Tiers

**Tier 1 — Intelligent Model Router**
A lightweight classifier that reads the incoming prompt and picks the best model from your pool. Runs locally on CPU in under 20ms.

- **SimilarityRouter**: Uses sentence-transformers to find similar past prompts and route to the model that worked best. Gets smarter with every request.
- **CostRouter**: Capability-aware routing with cost optimization. Respects `prefer_local` for air-gapped deployments.
- **EnsembleRouter**: Priority chain — tries similarity first, falls back to cost, then to default.

**Tier 2 — Multi-Agent Orchestrator** *(Phase 2)*
A planning model that decomposes hard tasks into subtasks, assigns them to specialist models, and synthesizes results. Uses reinforcement learning (GRPO-style) to learn teamwork patterns.

---

## Quickstart

### Install

```bash
git clone https://github.com/eulogik/fugusashi.git
cd fugusashi
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure

Edit `config.yaml` to add your models:

```yaml
default_model: "llama3.2-local"

models:
  - name: "llama3.2-local"
    provider: "ollama"
    model: "llama3.2:1b"
    api_base: "http://localhost:11434"
    cost_per_input_token: 0.0
    cost_per_output_token: 0.0
    capabilities: ["chat", "reasoning"]
    description: "Llama 3.2 1B (local, free)"

  - name: "gpt-4o-mini"
    provider: "openai"
    model: "gpt-4o-mini"
    cost_per_input_token: 0.00000015
    cost_per_output_token: 0.0000006
    capabilities: ["chat", "reasoning", "code", "creative"]
    description: "OpenAI GPT-4o-mini"
```

### Run

```bash
fugusashi serve --config config.yaml
# → Fugusashi router listening on 0.0.0.0:6060
```

### Use

```bash
# Auto-route
curl http://localhost:6060/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello"}]}'

# Force a specific model
curl http://localhost:6060/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2-local","messages":[{"role":"user","content":"Say hello"}]}'
```

---

## API

### `POST /v1/chat/completions`

OpenAI-compatible chat completion. Set `model: "auto"` for intelligent routing.

Every response includes a `routing_decision` field:

```json
{
  "id": "fugu-698f0a66db98",
  "model": "llama3.2-local",
  "choices": [...],
  "routing_decision": {
    "model": "llama3.2-local",
    "confidence": 0.9,
    "strategy": "ensemble(cost)",
    "latency_ms": 0.05,
    "explanation": "Routed by capability fit + cost. Top pick: llama3.2-local"
  }
}
```

### `GET /v1/models`

List available models with capabilities and pricing.

### `GET /v1/routing/decisions`

Recent routing decisions (configurable limit).

### `GET /v1/stats`

Aggregated stats: total requests, cost, tokens, per-model breakdown.

### `GET /v1/trace/{request_id}` 

Full trace for a specific request including routing + model calls.

### `POST /v1/routing/training`

Seed the similarity router with prompt→model preferences:

```bash
curl -X POST http://localhost:6060/v1/routing/training \
  -H "Content-Type: application/json" \
  -d '[{"prompt":"Write Python code","model":"gpt-4o-mini","score":0.95}]'
```

### `POST /v1/feedback/rate`

Rate a response quality (1-5). Feeds back into the router:

```bash
curl -X POST http://localhost:6060/v1/feedback/rate \
  -H "Content-Type: application/json" \
  -d '{"request_id":"fugu-698f0a66db98","rating":5}'
```

### `POST /v1/feedback/retrain`

Rebuild the similarity index from accumulated feedback:

```bash
curl -X POST http://localhost:6060/v1/feedback/retrain
```

### `GET /v1/feedback/stats`

Feedback statistics: outcomes, error rates, model rankings.

### `GET /v1/feedback/rankings`

Per-model win rates, costs, and user ratings.

---

## Dashboard

Open `http://localhost:6060/dashboard` for a live view:

- **Overview**: total requests, tokens, cost, avg routing latency
- **Model Usage**: bar chart of which models are being picked
- **Strategy Distribution**: cost vs similarity routing breakdown
- **Recent Decisions**: live table of every routing decision

Auto-refreshes every 3 seconds.

---

## Benchmarking

```bash
# Run default benchmark (20 samples)
fugusashi benchmark

# With training data
fugusashi benchmark --train --verbose

# JSON output
fugusashi benchmark --train --json

# Custom dataset
fugusashi benchmark --dataset my_data.jsonl --verbose
```

Custom dataset format (JSONL):
```jsonl
{"prompt":"How do I center a div?","expected_model":"gpt-4o-mini","category":"code"}
{"prompt":"What is 2+2?","expected_model":"llama3.2-local","category":"factual"}
```

---

## Feedback Loop

This is what makes Fugusashi better than Fugu. The system **learns from every request**:

1. **Route**: Router picks a model for the prompt
2. **Execute**: Model generates a response
3. **Evaluate**: Track outcome (success/failure, cost, latency)
4. **Learn**: Feed outcomes back into the similarity router

### Automatic learning

Outcomes are recorded automatically. When a model fails, that's recorded too. Over time, the similarity router builds a map of which prompts work best with which models.

### User feedback

Send ratings (1-5) to improve routing:

```bash
curl -X POST http://localhost:6060/v1/feedback/rate \
  -d '{"request_id":"<id>","rating":5}'
```

### Retraining

Trigger retraining when you have new feedback:

```bash
curl -X POST http://localhost:6060/v1/feedback/retrain
```

This rebuilds the similarity index from all accumulated outcomes, so the router gets smarter continuously.

---

## Project Structure

```
fugusashi/
├── config.yaml              # Model pool + routing config
├── pyproject.toml           # Dependencies + metadata
├── LIVING.md                # Living development walkthrough
├── src/fugusashi/
│   ├── __init__.py          # Package init
│   ├── __main__.py          # CLI: serve, benchmark
│   ├── server.py            # FastAPI app factory
│   ├── config.py            # Pydantic config from YAML
│   ├── providers.py         # LiteLLM multi-provider wrapper
│   ├── tracker.py           # Cost/routing transparency
│   ├── feedback.py          # Feedback loop + learning
│   ├── benchmark.py         # Benchmark runner
│   ├── api/
│   │   └── routes.py        # All API endpoints
│   ├── router/
│   │   ├── interface.py     # Abstract router protocol
│   │   ├── strategies.py        # Cost, Similarity, Fallback routers
│   │   └── ensemble.py      # Priority-chain ensemble
│   └── static/
│       └── dashboard.html   # Live web dashboard
└── tests/
    └── test_integration.py  # Integration tests
```

---

## How It Beats Fugu

1. **Transparent**: Every routing decision is visible and explainable
2. **Self-hosting**: Runs entirely on-premise with local models
3. **Learning**: Gets smarter from every request via feedback loop
4. **Open**: Community-curated preference datasets, not proprietary
5. **Extensible**: Add your own routing strategies via the plugin interface
6. **Observable**: Dashboard + stats + traces out of the box

---

## Roadmap

- [x] Tier 1: Intelligent model router (cost + similarity)
- [x] OpenAI-compatible API
- [x] Transparent routing decisions
- [x] Web dashboard
- [x] Benchmarking tool
- [x] Feedback loop with learning
- [ ] Tier 2: Multi-agent orchestrator
- [ ] GRPO-based workflow planning
- [ ] Community preference dataset sharing
- [ ] Plugin system for custom routers

---

## License

MIT
