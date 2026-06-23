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
def benchmark(dataset, threshold, verbose, json_out, train):
    from .benchmark import run_benchmark_cli
    run_benchmark_cli(dataset_path=dataset, threshold=threshold, verbose=verbose, json_out=json_out, train=train)


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
