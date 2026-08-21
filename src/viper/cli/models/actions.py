"""
Actions behind the model-listing CLI commands.

Framework-free readers over the ``viper.registry`` embedder registry, kept
directly unit-testable. Importing this module imports ``viper.models``, which
registers the built-in embedder factories.
"""

import viper.models  # noqa: F401  (populates the embedder factory registry)
from viper.registry import get_embedder_factory_registry, get_embedder_families


def list_model_keys() -> list[str]:
    """Returns all registered embedder keys, sorted."""
    return sorted(get_embedder_factory_registry().keys())


def list_model_families() -> dict[str, list[str]]:
    """Returns registered families mapped to their sorted member keys."""
    return get_embedder_families()
