---
language: en
license: mit
tags:
  - model-router
  - llm
  - fugusashi
  - preference-data
  - cma-es
  - routing
  - open-source
  - nlp
  - text-classification
  - prompt-engineering
task_categories:
  - text-classification
size_categories:
  - n<1K
library_name: datasets
pretty_name: Fugusashi Model Router Preferences
---

# Fugusashi Preferences — Model Router Training Dataset

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/fugusashi?color=ef4444&logo=pypi)](https://pypi.org/project/fugusashi/)
[![GitHub](https://img.shields.io/github/stars/eulogik/fugusashi?style=social)](https://github.com/eulogik/fugusashi)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/eulogik/fugusashi/blob/main/LICENSE)
[![Model](https://img.shields.io/badge/model-fugusashi--router-red.svg)](https://huggingface.co/eulogik/fugusashi-router)

</div>

## What is This?

Community-curated prompt → model preference dataset for **Fugusashi** router training. Each entry maps a prompt to the optimal model, used to train the CMA-ES evolved routing weights.

**Part of the [Fugusashi](https://github.com/eulogik/fugusashi) project — Like Sakana Fugu. But Free.**

> Note: this dataset contains 20 hand-curated seed samples. It is a starting point for community contributions, not a benchmark — sample sizes this small cannot support published benchmark claims.

## Dataset Format

JSONL with fields:

| Field | Type | Description |
|---|---|---|
| `prompt` | string | The user prompt |
| `preferred_model` | string | Optimal model for this prompt |
| `source` | string | Data source (e.g., `eulogik-seed`) |
| `category` | string | Prompt category (e.g., `code`, `creative`, `reasoning`) |
| `score` | float | Preference confidence (0-1) |

## Quick Load

```python
from datasets import load_dataset

dataset = load_dataset("eulogik/fugusashi-preferences")
for sample in dataset["train"]:
    print(f"{sample['prompt'][:50]}... → {sample['preferred_model']}")
```

## Data Sources

| Source | Samples | Description |
|---|---|---|
| eulogik-seed | 20 | Initial seed data curated by eulogik |

## How to Contribute

We welcome community contributions! To add your own preference samples:

1. Fork this dataset on HuggingFace
2. Add your entries in JSONL format
3. Submit a pull request

### Contribution Guidelines

- Prompts should be diverse (code, creative, reasoning, analysis)
- Models available: `gpt-oss-120b`, `nemotron-3-ultra`, `nemotron-3-super`, `hermes-3-405b`, `lfm-2.5-1.2b`
- Set `score` based on how confident you are in the preference (0.5-1.0)
- Include realistic prompts that users would actually send

## Project Links

| Resource | Link |
|---|---|
| Source Code | [github.com/eulogik/fugusashi](https://github.com/eulogik/fugusashi) |
| PyPI Package | [pypi.org/project/fugusashi](https://pypi.org/project/fugusashi/) |
| Live Demo | [huggingface.co/spaces/eulogik/fugusashi](https://huggingface.co/spaces/eulogik/fugusashi) |
| Router Model | [huggingface.co/eulogik/fugusashi-router](https://huggingface.co/eulogik/fugusashi-router) |
| Website | [eulogik.github.io/fugusashi](https://eulogik.github.io/fugusashi/) |
| eulogik | [eulogik.com](https://eulogik.com) |

## Citation

```bibtex
@dataset{fugusashi-preferences2026,
  title={Fugusashi Model Router Preferences},
  author={{eulogik}},
  year={2026},
  url={https://huggingface.co/datasets/eulogik/fugusashi-preferences}
}
```

## License

MIT — use it however you want.

---

<div align="center">

**Built with ❤️ by [eulogik](https://eulogik.com)**

*Fugu Sashi. Served Free.*

</div>
