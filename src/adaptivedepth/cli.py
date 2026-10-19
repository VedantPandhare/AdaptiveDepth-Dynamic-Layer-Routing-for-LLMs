from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from adaptivedepth.config import load_project_settings


app = typer.Typer(
    no_args_is_help=True,
    help="AdaptiveDepth verified-generation tools.",
)


@app.command("inspect-model")
def inspect_model(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/model.yaml"),
) -> None:
    """Load the configured model and report its discovered transformer stack."""

    from adaptivedepth.model.loading import load_model

    settings = load_project_settings(config)
    routed_model, _tokenizer, device = load_model(settings.model)
    typer.echo(
        json.dumps(
            {
                "model_id": settings.model.model_id,
                "device": str(device),
                "layer_path": routed_model.layer_path,
                "num_layers": routed_model.num_layers,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
