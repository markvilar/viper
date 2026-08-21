"""
Module for image embedder model registry.

Factories register under a flat string ``key`` (the identity used for lookup)
and carry a required ``family`` label that groups related keys — most immediately
the variants of a single model that differ only in configuration, such as the
SVD-truncated MegaLoc checkpoints (``megaloc``, ``megaloc-512d``, ...). ``family``
plays no role in lookup; it exists purely for enumeration.
"""

from collections.abc import Callable
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
