import gradio as gr
import numpy as np
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from fugusashi.coordinator import CMAESRouter, Task
from fugusashi.dataset import PreferenceDataset, seed_default_dataset
from fugusashi.federated import FederatedRouter, RoutingExplainer

# Initialize components
router = CMAESRouter(population_size=16, n_generations=30)
ds = PreferenceDataset(data_dir="/tmp/hf_space")
seed_default_dataset(ds)
tasks = [Task(p.prompt, p.category) for p in ds.preferences]
router.evolve(tasks, fast=True)
explainer = RoutingExplainer()


def route_prompt(prompt, strategy):
    if not prompt:
        return "Please enter a prompt."

    if strategy == "CMA-ES Coordinator":
        result = router.route(prompt)
        explanation = explainer.explain(prompt, result, result.scores)
        return explanation
    else:
        available = {
            "gpt-oss-120b": {"cost_per_input_token": 0, "cost_per_output_token": 0, "capabilities": ["chat", "code", "reasoning"]},
            "nemotron-3-ultra": {"cost_per_input_token": 0, "cost_per_output_token": 0, "capabilities": ["chat", "code", "reasoning"]},
            "hermes-3-405b": {"cost_per_input_token": 0, "cost_per_output_token": 0, "capabilities": ["chat", "reasoning", "creative"]},
            "lfm-2.5-1.2b": {"cost_per_input_token": 0, "cost_per_output_token": 0, "capabilities": ["chat", "code"]},
        }
        from fugusashi.router import EnsembleRouter
        ensemble = EnsembleRouter()
        result = ensemble.route(prompt, [{"role": "user", "content": prompt}], available)
        explanation = explainer.explain(prompt, type('obj', (), {'model': result.model, 'confidence': result.confidence, 'scores': result.scores, 'strategy': result.strategy, 'latency_ms': result.latency_ms})(), result.scores)
        return explanation


def get_stats():
    stats = router.get_stats()
    return f"""**CMA-ES Coordinator Stats**
- Generation: {stats['generation']}
- Best fitness: {stats['best_fitness']:.4f}
- Sigma: {stats['sigma']:.4f}
- Models: {', '.join(stats['model_names'])}
- Training tasks: {stats['history_length']}
"""


with gr.Blocks(
    title="Fugusashi Router Demo",
    theme=gr.themes.Soft(primary_hue="red"),
) as demo:
    gr.Markdown("# Fugusashi — Intelligent Model Router")
    gr.Markdown("*Like Sakana Fugu. But Free. And Yours.*")

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Write a Python function to sort a list...",
                lines=3,
            )
            strategy = gr.Radio(
                ["CMA-ES Coordinator", "Ensemble (Cost+Similarity)"],
                value="CMA-ES Coordinator",
                label="Routing Strategy",
            )
            route_btn = gr.Button("Route")

        with gr.Column():
            output = gr.Markdown(label="Routing Decision")

    route_btn.click(fn=route_prompt, inputs=[prompt, strategy], outputs=[output])

    with gr.Row():
        stats_btn = gr.Button("Show Coordinator Stats")
        stats_output = gr.Markdown()
    stats_btn.click(fn=get_stats, outputs=[stats_output])

    gr.Markdown("## Try these prompts")
    examples = [
        ["Write a Python class for a binary tree", "CMA-ES Coordinator"],
        ["What is 2+2?", "CMA-ES Coordinator"],
        ["Explain quantum entanglement simply", "CMA-ES Coordinator"],
        ["Write a bash script to backup files", "CMA-ES Coordinator"],
        ["Tell me a joke", "CMA-ES Coordinator"],
        ["Explain the theory of relativity", "CMA-ES Coordinator"],
    ]
    gr.Examples(examples=examples, inputs=[prompt, strategy])

    gr.Markdown("""
    ## API Usage

    ```bash
    pip install gradio_client
    from gradio_client import Client
    client = Client("eulogik/fugusashi")
    result = client.predict("Write a Python function", "CMA-ES Coordinator", api_name="/route_prompt")
    ```
    """)


demo.launch()
