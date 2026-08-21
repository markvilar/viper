"""
Top-level entrypoint for the viper CLI.

Defines the root ``viper`` command group and wires in its subgroups. The
``main`` function is the target of the ``viper`` console script declared in
``pyproject.toml``.
"""

import click

from viper.cli.forge import forge_group
from viper.cli.models import list_families, list_models


@click.group(name="viper")
def cli() -> None:
    """viper — tools for visual place recognition models."""


cli.add_command(forge_group, name="forge")
cli.add_command(list_models, name="list-models")
cli.add_command(list_families, name="list-families")


def main() -> None:
    """Console-script entrypoint for the viper CLI."""
    cli()
