"""
Module for image embedder model registry.

Factories register under a flat string ``key`` (the identity used for lookup)
and carry a required ``family`` label that groups related keys — most immediately
the variants of a single model that differ only in configuration, such as the
SVD-truncated MegaLoc checkpoints (``megaloc``, ``megaloc-512d``, ...). ``family``
plays no role in lookup; it exists purely for enumeration.
"""

from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from typing import Any
from viper.types import ImageEmbedder
from viper.types import ImageEmbedderFactory


_embedder_factories: dict[str, ImageEmbedderFactory] = dict()
_embedder_families: dict[str, str] = dict()


type FactoryRegistry = dict[str, ImageEmbedderFactory]


def register_embedder_factory(
    key: str,
    *,
    family: str,
) -> Callable[[ImageEmbedderFactory], ImageEmbedderFactory]:
    """
    Registers an image embedder factory.

    Arguments:
        key    - flat lookup key for the factory (e.g. "megaloc-512d")
        family - grouping label for related keys (e.g. "megaloc"); used only for
                 enumeration, never for lookup
    """

    def decorator(func: ImageEmbedderFactory) -> ImageEmbedderFactory:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ImageEmbedder:
            return func(*args, **kwargs)

        _embedder_factories[key] = wrapper
        _embedder_families[key] = family
        return wrapper

    return decorator


@dataclass(frozen=True)
class EmbedderRegistrationEntry:
    """A declarative registration for a remote-URL-backed embedder."""

    key: str
    """Flat lookup key for the variant (e.g. "megaloc-512d")."""

    family: str
    """Grouping label for related keys (e.g. "megaloc"); enumeration only."""

    checkpoint_url: str
    """URL the variant is built from (state-dict checkpoint, hub repo, ...)."""

    factory: Callable[[str], ImageEmbedder]
    """Builds the embedder from the URL; owns the load strategy end to end."""


def register_embedder_entries(entries: Iterable[EmbedderRegistrationEntry]) -> None:
    """Registers each entry as a standard (key, family) -> factory registration."""
    for entry in entries:

        @register_embedder_factory(entry.key, family=entry.family)
        def _factory_wrapper(
            entry: EmbedderRegistrationEntry = entry,
        ) -> ImageEmbedder:
            return entry.factory(entry.checkpoint_url)


def get_embedder_factory_registry() -> dict[str, ImageEmbedderFactory]:
    """Returns the embedder factory registry."""
    return _embedder_factories.copy()


def get_embedder_factory(key: str) -> ImageEmbedderFactory | None:
    """Returns the embedder factory for the given key."""
    return _embedder_factories.get(key)


def get_embedder_families() -> dict[str, list[str]]:
    """Returns the registered families mapped to their sorted member keys."""
    families: dict[str, list[str]] = dict()
    for key, family in _embedder_families.items():
        families.setdefault(family, []).append(key)
    for keys in families.values():
        keys.sort()
    return families
