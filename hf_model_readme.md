---
language: en
license: mit
tags:
  - model-router
  - llm
  - cma-es
  - open-source
  - fugusashi
datasets:
  - eulogik/fugusashi-preferences
---

# Fugusashi Router

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/fugusashi?color=ef4444&logo=pypi)](https://pypi.org/project/fugusashi/)
[![GitHub](https://img.shields.io/github/stars/eulogik/fugusashi?style=social)](https://github.com/eulogik/fugusashi)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/eulogik/fugusashi/blob/main/LICENSE)
[![Dataset](https://img.shields.io/badge/dataset-fugusashi--preferences-blue.svg)](https://huggingface.co/datasets/eulogik/fugusashi-preferences)
[![Space](https://img.shields.io/badge/demo-Live%20Space-orange.svg)](https://huggingface.co/spaces/eulogik/fugusashi)
[![Website](https://img.shields.io/badge/website-eulogik.github.io-red.svg)](https://eulogik.github.io/fugusashi/)

**By [eulogik](https://eulogik.com) — building AI infrastructure for everyone.**

</div>

---

## What is This?

CMA-ES evolved routing weights for the **Fugusashi** intelligent model router. A 385-dimensional weight vector that maps prompt embeddings to model selection — learned via Covariance Matrix Adaptation Evolution Strategy, the same approach used in Sakana AI's [TRINITY](https://arxiv.org/abs/2503.10018) paper.

**Like Sakana Fugu. But Free. And Yours.**

### What's New in v1.2.0

- **Multi-Agent Orchestrator** — decomposes complex tasks, assigns to specialist models, synthesizes results
- **GRPO Learning** — team reward scoring learns which model assignments work best
- **Auto-escalation** — Tier 1 automatically escalates to Tier 2 when confidence is low
- 6 new API endpoints for orchestration and learning

## Usage

```python
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load weights
with open("cmaes_weights.json") as f:
    data = json.load(f)

weights = np.array(data["mean"])
bias = data["mean"][-1]

# Embed prompt and route
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(["Write a Python function to sort a list"], normalize_embeddings=True)[0]

logits = embedding * weights[:384] + bias
models = ["gpt-oss-120b", "nemotron-3-ultra", "nemotron-3-super", "hermes-3-405b", "lfm-2.5-1.2b"]
chosen = models[np.argmax(logits)]
print(f"Route to: {chosen}")
```

## Training Details

| Field | Value |
|---|---|
| Algorithm | CMA-ES (Covariance Matrix Adaptation Evolution Strategy) |
| Dimensions | 385 (384 weights + 1 bias) |
| Population | 16 |
| Generations | 30 |
| Training tasks | 20 preference samples |
| Best fitness | ~0.24 |
| Embedding model | all-MiniLM-L6-v2 |

## Project Links

| Resource | Link |
|---|---|
| 🌐 Website | [eulogik.github.io/fugusashi](https://eulogik.github.io/fugusashi/) |
| 💻 Source Code | [github.com/eulogik/fugusashi](https://github.com/eulogik/fugusashi) |
| 📦 PyPI | [pypi.org/project/fugusashi](https://pypi.org/project/fugusashi/) |
| 📊 Dataset | [huggingface.co/datasets/eulogik/fugusashi-preferences](https://huggingface.co/datasets/eulogik/fugusashi-preferences) |
| 🚀 Live Demo | [huggingface.co/spaces/eulogik/fugusashi](https://huggingface.co/spaces/eulogik/fugusashi) |
| 🌍 eulogik | [eulogik.com](https://eulogik.com) |

## Citation

If you use Fugusashi in your work:

```
@software{fugusashi2026,
  title={Fugusashi: Open-Source Intelligent Model Router},
  author={{eulogik}},
  year={2026},
  url={https://github.com/eulogik/fugusashi}
}
```

## License

MIT — use it however you want.

---

<div align="center">**Built with ❤️ by [eulogik](https://eulogik.com)**</div>
