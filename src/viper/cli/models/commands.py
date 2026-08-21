"""
Root-level model-listing CLI commands.

Commands here own only the CLI surface; the work is delegated to
``viper.cli.models.actions``.
"""

import click

from viper.cli.console import console

from .actions import list_model_families, list_model_keys


@click.command(name="list-models")
def list_models() -> None:
    """List the registered embedder keys."""
    for key in list_model_keys():
        console.print(key)


@click.command(name="list-families")
def list_families() -> None:
    """List the registered model families and their member keys."""
    families = list_model_families()
    for family in sorted(families):
        keys = ", ".join(families[family])
        console.print(f"[bold]{family}[/bold]: {keys}")
