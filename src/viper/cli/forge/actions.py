"""
Actions behind the ``forge`` CLI commands.

These functions carry the work with no CLI-framework details: resolve a forge
from the ``viper.forge`` registry, load the source model through the
``viper.registry`` embedder factory, apply the forge, and write a
conventionally-named checkpoint. This keeps them directly unit-testable.

Importing this module imports ``viper.forge``, which registers the built-in
forges and, transitively, their embedder factories.
"""

from pathlib import Path

import viper.forge  # noqa: F401  (populates the forge and factory registries)
from viper.forge.registry import ForgeEntry, get_forge
from viper.registry import get_embedder_factory
from viper.types import ImageEmbedder


def resolve_forge(model_key: str, method: str) -> ForgeEntry:
    """
    Resolve the forge for a ``(model_key, method)`` pair.

    Raises:
        ValueError - if no forge is registered for the pair
    """
    entry = get_forge(model_key, method)
    if entry is None:
        raise ValueError(
            f"no forge registered for model '{model_key}' and method '{method}'"
        )
    return entry


def load_source_model(model_key: str) -> ImageEmbedder:
    """
    Load the pretrained source model by registry key.

    Inherits the device requirements of the resolved factory (the ``megaloc``
    factory requires CUDA).

    Raises:
        ValueError - if no embedder factory is registered for the key
    """
    factory = get_embedder_factory(model_key)
    if factory is None:
        raise ValueError(f"no embedder factory registered for model '{model_key}'")
    return factory()


def derive_checkpoint_name(model_key: str, dim: int, label: str, revision: str) -> str:
    """Derive the conventional checkpoint file name for a forged model."""
    return f"{model_key}-{dim}d-{label}-v{revision}.pth"


def adapt_model(
    model_key: str,
    method: str,
    dim: int,
    revision: str = "1.0",
    output: Path | None = None,
) -> Path:
    """
    Forge a model and write its checkpoint to disk.

    Resolves the forge, loads the source model, applies the forge with the
    target dimension, moves the result to CPU, and serialises its ``state_dict``
    so the checkpoint round-trips through the ``build_<model>_from_state_dict``
    builders.

    Arguments:
        model_key - embedder registry key of the source model
        method    - decomposition method selector
        dim       - target descriptor dimension
        revision  - version component of the derived checkpoint name
        output    - explicit output path; otherwise a name is derived by
                    convention in the current working directory
    Returns:
        the path the checkpoint was written to
    """
    import torch

    entry = resolve_forge(model_key, method)
    model = load_source_model(model_key)
    forged = entry.forge(model, dim)
    forged = forged.to("cpu")

    if output is None:
        name = derive_checkpoint_name(model_key, dim, entry.label, revision)
        output = Path.cwd() / name

    torch.save(forged.state_dict(), output)
    return output
