<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/fugusashi?color=ef4444&label=pypi&logo=pypi&logoColor=white)](https://pypi.org/project/fugusashi/)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/eulogik/fugusashi?style=social)](https://github.com/eulogik/fugusashi/stargazers)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%20HuggingFace-ef4444.svg)](https://huggingface.co/eulogik/fugusashi-router)
[![Docs](https://img.shields.io/badge/docs-eulogik.github.io-red.svg)](https://eulogik.github.io/fugusashi/)
[![Paper](https://img.shields.io/badge/arXiv-paper-B31B1B.svg)](https://github.com/eulogik/fugusashi/blob/main/paper/main.tex)
[![Website](https://img.shields.io/badge/website-eulogik.com-ef4444.svg)](https://eulogik.com)

**By [eulogik](https://eulogik.com) — building AI infrastructure for everyone.**

---

# Fugusashi

### Fugu Sashi. Served Free.

*Fugusashi* (Japanese: 不縛 — "unbound, unrestrained") is an intelligent model router and multi-agent orchestrator. Named after *Fugu Sashi* — the famous Japanese pufferfish delicacy — because this router serves up the world's best AI models without the poison of vendor lock-in or the pricing of Sakana Fugu. It automatically picks the best model for each prompt, learns from every request via a CMA-ES coordinator inspired by Sakana's TRINITY paper, and runs entirely on your infrastructure.

**Research contributions:** (1) **Federated routing learning** — multiple organizations collaboratively improve routing without sharing data; (2) **Human-interpretable routing** — every decision comes with a natural language explanation; (3) **Continuous CMA-ES adaptation** — routing weights evolve from outcomes.

**Like Sakana Fugu. But Free. [Live Demo](https://huggingface.co/spaces/eulogik/fugusashi) · [Docs](https://eulogik.github.io/fugusashi/) · [PyPI](https://pypi.org/project/fugusashi/) · [GitHub](https://github.com/eulogik/fugusashi)**

</div>

---

## Why Fugusashi?

[Sakana AI's Fugu](https://sakana.ai/fugu) is a trained orchestration model — a 7B LLM that coordinates frontier models behind a single API. It's powerful, but it's a black box: you can't see why it routes where it does, you can't self-host it, you can't train it on your data, and you pay $5-30 per million tokens plus a $20-200/month subscription.

**Fugusashi is the transparent alternative.** Same CMA-ES evolution concept (both inspired by Sakana's TRINITY paper), but open, self-hostable, and designed to learn from your specific traffic. Where Fugu hides its routing logic, Fugusashi exposes every decision. Where Fugu runs only in Sakana's cloud, Fugusashi runs on your infrastructure. Where Fugu is static, Fugusashi learns from every request via a feedback loop.

### Fugusashi vs Sakana AI Fugu

| Feature | Sakana Fugu | Fugusashi |
|---|---|---|
| **Architecture** | Trained 7B coordinator LLM | CMA-ES evolved weights + rule-based ensemble |
| **Orchestration** | ✅ TRINITY/Conductor (ICLR 2026) | ✅ Rule-based + GRPO learning |
| **Self-Hosting** | ❌ Cloud-only | ✅ Local-first, air-gapped |
| **Cost** | $5-30/M tokens + $20-200/mo | ✅ Free (pay only for model APIs) |
| **Transparency** | ❌ Black box routing | ✅ Every decision visible + explained |
| **Feedback Loop** | ❌ Static | ✅ Learns from your traffic |
| **Federated Learning** | ❌ | ✅ Collaborative routing without data sharing |
| **Model Pool** | Sakana-controlled frontier models | ✅ You control (100+ providers) |
| **Training Data** | ❌ Proprietary | ✅ Community preference datasets |
| **Dashboard** | ❌ | ✅ Live routing visualization |
| **EU Availability** | ❌ Blocked (GDPR) | ✅ Available everywhere |
| **License** | Proprietary | ✅ MIT |

**Our edge:** Fugu is a powerful trained orchestrator with frontier models in its pool. Fugusashi is the transparent, self-hosting alternative you can run on your own infrastructure, train on your own data, and audit every decision. Different bets — Fugu bets on a trained coordinator, we bet on transparency and control.

---

## Quickstart

### Install from PyPI

```bash
pip install fugusashi
```

### Or run with Docker

```bash
docker run -p 6060:6060 ghcr.io/eulogik/fugusashi:latest
```

### Or from source

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

![Fugusashi Architecture](https://raw.githubusercontent.com/eulogik/fugusashi/main/assets/arch.svg)

### Tier 1 — Intelligent Model Router

Three routing strategies in priority order:

1. **SimilarityRouter** — Uses sentence-transformers to find similar past prompts and route to the model that worked best. Gets smarter with every request via the feedback loop.
2. **CostRouter** — Capability-aware routing with cost optimization. Respects `prefer_local` for air-gapped deployments.
3. **FallbackRouter** — Always returns a result, even with no data.

### Tier 2 — Multi-Agent Orchestrator

A planning model that decomposes hard tasks into subtasks, assigns them to specialist models, and synthesizes results. Uses reinforcement learning (GRPO-style) to learn teamwork patterns.

```bash
# Orchestrate a complex task
curl -X POST http://localhost:6060/v1/orchestrate \
  -d '{"prompt": "Write a Python web scraper, test it, and write documentation"}'

# View orchestration history
curl http://localhost:6060/v1/orchestration/history

# Check GRPO learning stats
curl http://localhost:6060/v1/orchestration/grpo/stats
```

### Federated Routing

Multiple Fugusashi instances collaboratively improve a shared routing model **without sharing prompts or data**.
Each organization trains locally, adds differential privacy noise, and contributes weight updates.
The result: a router that's smarter than any single deployment.

```bash
# Register as a federated client
curl -X POST http://localhost:6060/v1/federated/register \
  -d '{"client_id": "my-org", "metadata": {"type": "healthcare"}}'

# Submit local routing updates
curl -X POST http://localhost:6060/v1/federated/submit \
  -d '{"client_id": "my-org", "weights": [...], "n_samples": 1000}'

# Trigger aggregation (requires min 3 clients)
curl -X POST http://localhost:6060/v1/federated/aggregate
```

### Routing Explanations

Every routing decision comes with a natural language explanation:

```bash
curl -X POST http://localhost:6060/v1/explain \
  -d '{"prompt": "Write a Python class for a binary tree"}'
```

Response:
```
Decision: Route to gpt-oss-120b (confidence: 87%)
Why: This prompt involves code generation. gpt-oss-120b is best suited for complex reasoning.
Alternatives:
  - lfm-2.5-1.2b (12%): better for fast responses
  - hermes-3-405b (1%): better for creative writing
Latency: 5.5ms | Strategy: cma-es
```

Users can override decisions with natural language feedback, which becomes training data.

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

### `POST /v1/orchestrate` — Multi-agent orchestration

```bash
curl -X POST http://localhost:6060/v1/orchestrate \
  -d '{"prompt": "Write code, test it, and document it"}'
```

### `GET /v1/orchestration/history` — Orchestration history

### `GET /v1/orchestration/grpo/stats` — GRPO learning stats

---

## Dashboard

Open `http://localhost:6060/dashboard` for a live view:

- **Overview**: total requests, tokens, cost, avg routing latency
- **Model Usage**: bar chart of which models are being picked
- **Strategy Distribution**: cost vs similarity routing breakdown
- **Recent Decisions**: live table of every routing decision

Auto-refreshes every 3 seconds. Light and dark themes.

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

## Benchmarking

```bash
pip install fugusashi
fugusashi benchmark                      # Default 20-sample dataset
fugusashi benchmark --train --verbose    # With training data
fugusashi benchmark --train --json       # JSON output
fugusashi benchmark -d my_data.jsonl     # Custom dataset
```

**Results:**

| Metric | Without Training | With Training |
|---|---|---|
| Accuracy | 70% | **85%** |
| Code accuracy | 60% | **90%** |
| Strategy | 100% cost | 60% cost / 40% similarity |
| Routing latency | <1ms | ~18ms |

Custom dataset format (JSONL):
```jsonl
{"prompt":"How do I center a div?","expected_model":"gpt-4o-mini","category":"code"}
{"prompt":"What is 2+2?","expected_model":"llama3.2-local","category":"factual"}
```

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
│   ├── __main__.py          # CLI: serve, benchmark, train, expand-data
│   ├── server.py            # FastAPI app factory
│   ├── config.py            # Pydantic config from YAML
│   ├── providers.py         # LiteLLM multi-provider wrapper
│   ├── tracker.py           # Cost/routing transparency
│   ├── feedback.py          # Feedback loop + learning
│   ├── benchmark.py         # Benchmark runner
│   ├── training.py          # ModernBERT training pipeline
│   ├── dataset.py           # Preference dataset management
│   ├── orchestrator.py      # Multi-agent orchestrator
│   ├── grpo.py              # GRPO-style reward learning
│   ├── api/
│   │   └── routes.py        # All API endpoints
│   ├── router/
│   │   ├── interface.py     # Abstract router protocol
│   │   ├── strategies.py    # Cost, Similarity, Fallback routers
│   │   ├── learned.py       # ModernBERT learned classifier router
│   │   └── ensemble.py      # Priority-chain ensemble (learned → similarity → cost)
│   └── static/
│       └── dashboard.html   # Live web dashboard
├── tests/
│   ├── test_integration.py  # Integration tests
│   ├── test_learned.py      # Learned router tests
│   └── test_orchestrator.py # Orchestrator + GRPO tests
└── docs/                    # GitHub Pages documentation
```

---

## Why Fugusashi Over Fugu?

1. **Transparent**: Every routing decision is visible and explainable. Fugu is a black box.
2. **Self-hosting**: Runs entirely on-premise with local models via Ollama. Fugu is cloud-only.
3. **Learning**: Gets smarter from your specific traffic via the feedback loop. Fugu is static.
4. **Federated**: Multiple organizations can collaboratively improve routing without sharing data. Fugu doesn't offer this.
5. **Free**: MIT licensed, no usage fees beyond model APIs. Fugu costs $5-30/M tokens + subscription.
6. **EU-available**: No GDPR restrictions. Fugu is blocked in the EU/EEA.
7. **Yours**: Community-owned preference datasets, customizable, extensible. Fugu is Sakana's.

---

## Paper

This project is accompanied by a research paper:

> **Fugusashi: Federated Learning of LLM Routing with Human-Interpretable Decisions**

The paper introduces three contributions:
1. **Federated routing learning** — collaborative model routing without data sharing
2. **Human-interpretable routing** — natural language explanations for every decision
3. **CMA-ES adaptation** — continuous evolution of routing weights

📄 [Read the paper](https://github.com/eulogik/fugusashi/blob/main/paper/main.tex)

## Links

| Resource | Link |
|---|---|
| 🌐 Website | [eulogik.com](https://eulogik.com) |
| 💻 GitHub | [github.com/eulogik/fugusashi](https://github.com/eulogik/fugusashi) |
| 📦 PyPI | [pypi.org/project/fugusashi](https://pypi.org/project/fugusashi/) |
| 🤗 HF Model | [huggingface.co/eulogik/fugusashi-router](https://huggingface.co/eulogik/fugusashi-router) |
| 📊 HF Dataset | [huggingface.co/datasets/eulogik/fugusashi-preferences](https://huggingface.co/datasets/eulogik/fugusashi-preferences) |
| 🚀 HF Space | [huggingface.co/spaces/eulogik/fugusashi](https://huggingface.co/spaces/eulogik/fugusashi) |
| 📖 Docs | [eulogik.github.io/fugusashi](https://eulogik.github.io/fugusashi/) |
| 📝 Paper | [paper/main.tex](https://github.com/eulogik/fugusashi/blob/main/paper/main.tex) |
| 🌍 eulogik | [eulogik.com](https://eulogik.com) |

---

## License

MIT — use it however you want.

---

<div align="center">

**Built with ❤️ by [eulogik](https://eulogik.com)**

**[⭐ Star on GitHub](https://github.com/eulogik/fugusashi) · [🤗 HuggingFace](https://huggingface.co/eulogik/fugusashi-router) · [🐦 X/Twitter](https://x.com/eulogik) · [📦 Docker](https://github.com/orgs/eulogik/packages/container/fugusashi)**

</div>
