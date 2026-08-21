"""
Tests for the forge CLI: the action layer (forge resolution, name derivation,
checkpoint writing) and the click command wiring.

The real ``megaloc`` factory requires a GPU and a checkpoint download, so these
tests register lightweight dummy factories and forges under a test key. This
seams out the CUDA-bound load while exercising the full dispatch and IO path on
CPU.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from click.testing import CliRunner

from viper.cli.entrypoint import cli
from viper.cli.forge import actions
from viper.forge.registry import _forges, register_forge
from viper.registry import _embedder_factories, register_embedder_factory

_TEST_MODEL_KEY = "dummy"
_TEST_METHOD = "svd"


@pytest.fixture
def registered_dummy() -> None:
    # Arrange a dummy factory + forge, cleaned up afterwards.
    @register_embedder_factory(key=_TEST_MODEL_KEY)
    def _load_dummy() -> nn.Module:
        return nn.Linear(8, 16)

    @register_forge(
        model_key=_TEST_MODEL_KEY, method=_TEST_METHOD, label="svd-truncated"
    )
    def _forge_dummy(model: nn.Module, k: int) -> nn.Module:
        return nn.Linear(model.in_features, k)

    yield

    _embedder_factories.pop(_TEST_MODEL_KEY, None)
    _forges.pop((_TEST_MODEL_KEY, _TEST_METHOD), None)


def test_derive_checkpoint_name_follows_convention() -> None:
    # Act
    name = actions.derive_checkpoint_name(
        model_key="megaloc", dim=512, label="svd-truncated", revision="1.0"
    )

    # Assert
    assert name == "megaloc-512d-svd-truncated-v1.0.pth"


def test_resolve_forge_raises_for_unregistered_pair() -> None:
    # Act / Assert
    with pytest.raises(ValueError, match="no forge registered"):
        actions.resolve_forge("nonexistent", "svd")


def test_adapt_model_writes_roundtrippable_checkpoint(
    registered_dummy: None, tmp_path: Path
) -> None:
    # Arrange
    output = tmp_path / "out.pth"

    # Act
    path = actions.adapt_model(
        _TEST_MODEL_KEY, method=_TEST_METHOD, dim=4, output=output
    )

    # Assert
    assert path == output
    state_dict = torch.load(output, weights_only=True)
    assert state_dict["weight"].shape == (4, 8)


def test_adapt_model_derives_name_when_output_omitted(
    registered_dummy: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)

    # Act
    path = actions.adapt_model(_TEST_MODEL_KEY, method=_TEST_METHOD, dim=4)

    # Assert
    assert path == tmp_path / "dummy-4d-svd-truncated-v1.0.pth"
    assert path.exists()


def test_adapt_command_wires_options(registered_dummy: None, tmp_path: Path) -> None:
    # Arrange
    output = tmp_path / "cli.pth"
    runner = CliRunner()

    # Act
    result = runner.invoke(
        cli,
        ["forge", "adapt", _TEST_MODEL_KEY, "--dim", "4", "--output", str(output)],
    )

    # Assert
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "Wrote" in result.output


def test_adapt_command_reports_unregistered_model() -> None:
    # Arrange
    runner = CliRunner()

    # Act
    result = runner.invoke(cli, ["forge", "adapt", "nonexistent", "--dim", "4"])

    # Assert
    assert result.exit_code != 0
    assert "no forge registered" in result.output
