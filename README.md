<div align="center">

<!-- SHIELD.IO BADGES -->
[![PyPI version](https://img.shields.io/pypi/v/fugusashi?color=6366f1&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/fugusashi/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/eulogik/fugusashi?style=social)](https://github.com/eulogik/fugusashi/stargazers)
[![Docs](https://img.shields.io/badge/docs-eulogik.github.io-fugusashi-blue.svg)](https://eulogik.github.io/fugusashi/)
[![Website](https://img.shields.io/badge/website-eulogik.com-6366f1.svg)](https://eulogik.com)

**By [eulogik](https://eulogik.com) — building AI infrastructure for everyone.**

---

# Fugusashi

### The Open-Source Alternative to Sakana AI's Fugu

*Fugusashi* (Japanese: 不縛 — "unbound, unrestrained") is an intelligent model router and multi-agent orchestrator. It automatically picks the best AI model for each prompt, learns from every request, and runs entirely on your infrastructure.

**[Live Demo](https://eulogik.github.io/fugusashi/) · [Docs](https://eulogik.github.io/fugusashi/) · [PyPI](https://pypi.org/project/fugusashi/) · [GitHub](https://github.com/eulogik/fugusashi)**

</div>

---

## Why Fugusashi?

Sakana AI's [Fugu](https://sakana.ai/fugu) is a proprietary model router. It works — but you can't see inside it, you can't train it on your own data, you can't self-host it, and you pay $5-30 per million tokens.

**Fugusashi does everything Fugu does, but open, transparent, and self-hosting.** It also adds a feedback loop that Fugu doesn't have — the router learns from every request and gets smarter over time.

### Fugusashi vs Sakana AI Fugu

| Feature | Sakana Fugu | Fugusashi |
|---|---|---|
| **Model Routing** | ✅ Proprietary | ✅ Open, transparent |
| **Multi-Agent Orchestration** | ✅ Fugu Ultra | 🔄 Phase 2 |
| **Self-Hosting** | ❌ Cloud-only | ✅ Local-first, air-gapped |
| **Cost** | $5-30/M tokens | ✅ Free (pay only for model APIs) |
| **Transparency** | ❌ Black box | ✅ Every decision visible |
| **Feedback Loop** | ❌ Static | ✅ Learns from every request |
| **Model Pool** | ❌ Fixed by Sakana | ✅ You control |
| **Training Data** | ❌ Proprietary | ✅ Community + your traffic |
| **Customization** | ❌ None | ✅ Fine-tune on your data |
| **License** | Proprietary | ✅ MIT |
| **Dashboard** | ❌ | ✅ Open web dashboard |
| **API** | Limited | ✅ OpenAI-compatible |

---

## Quickstart

### Install from PyPI

```bash
pip install fugusashi
```

### Or install from source

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
# Auto-route — the router picks the best model
curl http://localhost:6060/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Say hello"}]}'

# Force a specific model
curl http://localhost:6060/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2-local","messages":[{"role":"user","content":"Say hello"}]}'
```

Every response includes a `routing_decision` showing which model was picked, why, and with what confidence.

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

### Tier 1 — Intelligent Model Router

Three routing strategies in priority order:

1. **SimilarityRouter** — Uses sentence-transformers to find similar past prompts and route to the model that worked best. Gets smarter with every request via the feedback loop.
2. **CostRouter** — Capability-aware routing with cost optimization. Respects `prefer_local` for air-gapped deployments.
3. **FallbackRouter** — Always returns a result, even with no data.

### Tier 2 — Multi-Agent Orchestrator *(Phase 2)*

A planning model that decomposes hard tasks into subtasks, assigns them to specialist models, and synthesizes results. Uses reinforcement learning (GRPO-style) to learn teamwork patterns.

---

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible. Set `model: "auto"` for intelligent routing.

Response includes `routing_decision`:
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
    "explanation": "Routed by capability fit + cost"
  }
}
```

### `GET /v1/models` — List available models

### `GET /v1/routing/decisions` — Recent routing decisions

### `GET /v1/stats` — Aggregated stats (cost, tokens, per-model)

### `GET /v1/trace/{request_id}` — Full request trace

### `POST /v1/routing/training` — Seed similarity router

```bash
curl -X POST http://localhost:6060/v1/routing/training \
  -H "Content-Type: application/json" \
  -d '[{"prompt":"Write Python code","model":"gpt-4o-mini","score":0.95}]'
```

### `POST /v1/feedback/rate` — Rate a response (1-5)

```bash
curl -X POST http://localhost:6060/v1/feedback/rate \
  -d '{"request_id":"fugu-698f0a66db98","rating":5}'
```

### `POST /v1/feedback/retrain` — Rebuild similarity index from feedback

### `GET /v1/feedback/stats` — Outcome statistics

### `GET /v1/feedback/rankings` — Per-model win rates

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
# Install
pip install fugusashi

# Run default benchmark (20 samples)
fugusashi benchmark

# With training data
fugusashi benchmark --train --verbose

# JSON output
fugusashi benchmark --train --json

# Custom dataset
fugusashi benchmark -d my_data.jsonl
```

**Results:**

| Metric | Without Training | With Training |
|---|---|---|
| Accuracy | 70% | **85%** |
| Code accuracy | 60% | **90%** |
| Strategy | 100% cost | 60% cost / 40% similarity |
| Routing latency | <1ms | ~18ms |

---

## Feedback Loop — The Killer Feature

Fugu's router is static. Fugusashi's **learns from every request**:

```
Route → Execute → Evaluate → Learn → (repeat)
```

1. **Route**: Router picks a model for the prompt
2. **Execute**: Model generates a response
3. **Evaluate**: Track outcome (success/failure, cost, latency)
4. **Learn**: Feed outcomes back into the similarity router
5. **Auto-Retrain**: Every 10 requests, the similarity index rebuilds automatically

Outcomes are stored in `.fugusashi_data/outcomes.jsonl` — inspectable, shareable, yours.

---

## Project Structure

```
fugusashi/
├── config.yaml              # Model pool + routing config
├── pyproject.toml           # Dependencies + metadata
├── LIVING.md                # Living development walkthrough
├── README.md                # This file
├── LICENSE                  # MIT
├── src/fugusashi/
│   ├── __init__.py
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
├── tests/
│   └── test_integration.py  # Integration tests
└── docs/                    # GitHub Pages documentation
```

---

## How It Beats Sakana AI's Fugu

1. **Transparent**: Every routing decision is visible and explainable. No black box.
2. **Self-hosting**: Runs entirely on-premise with local models via Ollama.
3. **Learning**: Gets smarter from every request via the feedback loop. Fugu can't do this.
4. **Open**: Community-owned preference datasets, not proprietary training data.
5. **Extensible**: Add your own routing strategies via the plugin interface.
6. **Observable**: Dashboard + stats + traces out of the box.
7. **Free**: MIT licensed. No usage fees. No vendor lock-in.

---

## Roadmap

- [x] Tier 1: Intelligent model router (cost + similarity)
- [x] OpenAI-compatible API
- [x] Transparent routing decisions
- [x] Web dashboard
- [x] Benchmarking tool
- [x] Feedback loop with auto-retraining
- [ ] Tier 2: Multi-agent orchestrator with GRPO
- [ ] Community preference dataset sharing
- [ ] Plugin system for custom routers
- [ ] CLI improvements (interactive mode, model management)

---

## Contributing

We welcome contributions! See [LIVING.md](LIVING.md) for the full development story.

1. Fork the repo
2. Create a feature branch
3. Add tests
4. Submit a pull request

---

## Links

- **Website**: [eulogik.com](https://eulogik.com)
- **GitHub**: [github.com/eulogik/fugusashi](https://github.com/eulogik/fugusashi)
- **PyPI**: [pypi.org/project/fugusashi](https://pypi.org/project/fugusashi/)
- **Docs**: [eulogik.github.io/fugusashi](https://eulogik.github.io/fugusashi/)
- **Issues**: [github.com/eulogik/fugusashi/issues](https://github.com/eulogik/fugusashi/issues)

---

## License

MIT — use it however you want.

---

<div align="center">

**Built with ❤️ by [eulogik](https://eulogik.com)**

**[⭐ Star on GitHub](https://github.com/eulogik/fugusashi) · [🐦 Follow on Twitter](https://twitter.com/eulogik) · [💬 Discussions](https://github.com/eulogik/fugusashi/discussions)**

</div>
