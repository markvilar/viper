"""
The ``forge`` CLI subgroup: parsing and wiring for model-forging commands.

Commands here own only the CLI surface; the work is delegated to
``viper.cli.forge.actions``.
"""

from pathlib import Path

import click

from viper.cli.console import console

from .actions import adapt_model


@click.group(name="forge")
def forge_group() -> None:
    """Forge decomposed variants of pretrained embedders."""


@forge_group.command(name="adapt")
@click.argument("model_key", metavar="MODEL_KEY")
@click.option(
    "--method",
    default="svd",
    show_default=True,
    help="Decomposition method to apply.",
)
@click.option(
    "--dim",
    type=int,
    required=True,
    help="Target descriptor dimension.",
)
@click.option(
    "--revision",
    default="1.0",
    show_default=True,
    help="Version component of the derived checkpoint name.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Explicit output path; overrides the derived checkpoint name.",
)
def adapt(
    model_key: str,
    method: str,
    dim: int,
    revision: str,
    output: Path | None,
) -> None:
    """Forge MODEL_KEY into a lower-dimensional variant and write a checkpoint."""
    try:
        path = adapt_model(
            model_key,
            method=method,
            dim=dim,
            revision=revision,
            output=output,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    console.print(f"Wrote [bold]{path}[/bold]")
