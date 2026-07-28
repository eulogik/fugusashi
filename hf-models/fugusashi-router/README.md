---
library_name: transformers
tags:
  - fugusashi
  - model-router
  - modernbert
  - eulogik
  - llm-routing
  - classifier
  - llm-orchestration
license: mit
---

# Fugusashi Router — ModernBERT Classifier

Fine-tuned ModernBERT-base (149M params) for intelligent LLM routing and dispatch.

## v1.3.0

- **80.0% test accuracy** (179 train / 45 test) across 3 model classes
- **83.3% benchmark accuracy** on held-out prompts
- ModernBERT + CMA-ES + ensemble routing
- 83ms inference on CPU, 137s training time
- Federated learning with differential privacy
- Human-readable explanations with override

## Links

- [GitHub](https://github.com/eulogik/fugusashi)
- [PyPI](https://pypi.org/project/fugusashi/)
- [Live Demo](https://huggingface.co/spaces/eulogik/fugusashi)
- [Full v1.3.0 Model Card](https://huggingface.co/eulogik/fugusashi-v1.3)
