from __future__ import annotations

import sys
from pathlib import Path

import click
import uvicorn


@click.group()
def cli():
    pass


@cli.command()
@click.option("--dataset", "-d", type=click.Path(exists=True), help="JSONL dataset file")
@click.option("--threshold", "-t", default=0.4, type=float, help="Confidence threshold")
@click.option("--verbose", "-v", is_flag=True, help="Show per-sample results")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
@click.option("--train", is_flag=True, help="Seed training data for similarity routing")
@click.option("--model-dir", default=".fugusashi_data/router_model", help="Trained model directory")
@click.option("--no-learned", is_flag=True, help="Disable learned router (baseline comparison)")
def benchmark(dataset, threshold, verbose, json_out, train, model_dir, no_learned):
    from .benchmark import run_benchmark_cli
    run_benchmark_cli(
        dataset_path=dataset, threshold=threshold, verbose=verbose,
        json_out=json_out, train=train, model_dir=model_dir, no_learned=no_learned,
    )


@cli.command()
@click.option("--data-dir", default=".fugusashi_data", help="Data directory")
def expand_data(data_dir):
    """Expand the training dataset with seed data + synthetic variants."""
    from .training import expand_dataset
    count = expand_dataset(data_dir=data_dir)
    click.echo(f"Expanded dataset to {count} examples in {data_dir}/expanded_preferences.jsonl")


@cli.command()
@click.option("--data-dir", default=".fugusashi_data", help="Data directory")
@click.option("--model-dir", default=".fugusashi_data/router_model", help="Output directory")
@click.option("--epochs", default=6, type=int, help="Training epochs")
@click.option("--lr", default=5e-5, type=float, help="Learning rate")
def train(data_dir, model_dir, epochs, lr):
    """Fine-tune the ModernBERT learned router on preference data."""
    from .training import TrainingConfig, train_modernbert
    config = TrainingConfig(epochs=epochs, learning_rate=lr)
    result = train_modernbert(model_dir=model_dir, data_dir=data_dir, config=config)
    click.echo(f"\nTraining complete in {result.training_time_ms:.0f}ms")
    click.echo(f"  Accuracy:  {result.accuracy:.2%}")
    click.echo(f"  Top-3:     {result.top3_accuracy:.2%}")
    click.echo(f"  Classes:   {result.n_classes} ({', '.join(result.per_class_accuracy.keys())})")
    click.echo(f"  Train:     {result.n_train} / Test: {result.n_test}")
    click.echo(f"  Savings:   {result.cost_savings:.0f}% routed to free models")
    click.echo(f"  Model:     {result.backbone}")
    click.echo(f"  Saved to:  {result.model_path}")


@cli.command()
@click.option("--config", "-c", default="config.yaml", help="Path to config file")
@click.option("--host", default=None, help="Bind address")
@click.option("--port", default=None, type=int, help="Bind port")
@click.option("--reload", is_flag=True, help="Auto-reload on file changes")
def serve(config: str, host: str | None, port: int | None, reload: bool):
    from .config import AppConfig

    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"Config file not found: {config}", err=True)
        sys.exit(1)

    cfg = AppConfig.from_yaml(str(config_path))

    from .server import create_app

    app = create_app(cfg)

    click.echo(
        f" Fugusashi router listening on "
        f"{host or cfg.host}:{port or cfg.port}"
    )

    uvicorn.run(
        app,
        host=host or cfg.host,
        port=port or cfg.port,
        reload=reload,
        log_level=cfg.observability.log_level.lower(),
    )


@cli.command()
def version():
    from . import __version__
    click.echo(f"fugusashi v{__version__}")


if __name__ == "__main__":
    cli()
