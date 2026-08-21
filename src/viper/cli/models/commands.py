"""
The ``models`` CLI subgroup: listing registered embedders and their families.

Commands here own only the CLI surface; the work is delegated to
``viper.cli.models.actions``.
"""

import click

from viper.cli.console import console

from .actions import list_model_families, list_model_keys


@click.group(name="models")
def models_group() -> None:
    """Inspect the registered embedder models."""


@models_group.command(name="list")
def list_models() -> None:
    """List the registered embedder keys."""
    for key in list_model_keys():
        console.print(key)


@models_group.command(name="families")
def list_families() -> None:
    """List the registered model families and their member keys."""
    families = list_model_families()
    for family in sorted(families):
        keys = ", ".join(families[family])
        console.print(f"[bold]{family}[/bold]: {keys}")
