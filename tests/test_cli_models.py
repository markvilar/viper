"""
Tests for the model-listing CLI actions and commands.
"""

from click.testing import CliRunner

from viper.cli.entrypoint import cli
from viper.cli.models import actions


def test_list_model_keys_includes_megaloc_variants() -> None:
    keys = actions.list_model_keys()
    assert keys == sorted(keys)
    for key in ("megaloc", "megaloc-256d", "megaloc-512d", "megaloc-1024d"):
        assert key in keys


def test_megaloc_variants_share_one_family() -> None:
    families = actions.list_model_families()
    assert families["megaloc"] == [
        "megaloc",
        "megaloc-1024d",
        "megaloc-256d",
        "megaloc-512d",
    ]


def test_models_list_command_lists_keys() -> None:
    result = CliRunner().invoke(cli, ["models", "list"])
    assert result.exit_code == 0
    assert "megaloc-512d" in result.output


def test_models_families_command_groups_keys() -> None:
    result = CliRunner().invoke(cli, ["models", "families"])
    assert result.exit_code == 0
    assert "megaloc" in result.output
    assert "megaloc-512d" in result.output
