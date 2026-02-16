"""
Module for image embedder model registry.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any
from viper.types import ImageEmbedder
from viper.types import ImageEmbedderFactory


_embedder_factories: dict[str, ImageEmbedderFactory] = dict()


type FactoryRegistry = dict[str, ImageEmbedderFactory]


def register_embedder_factory(
    key: str,
) -> Callable[[ImageEmbedderFactory], ImageEmbedderFactory]:
    """Registers an image embedder factory."""

    def decorator(func: ImageEmbedderFactory) -> ImageEmbedderFactory:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> ImageEmbedder:
            return func(*args, **kwargs)

        _embedder_factories[key] = wrapper
        return wrapper

    return decorator


def get_embedder_factory_registry() -> dict[str, ImageEmbedderFactory]:
    """Returns the embedder factory registry."""
    return _embedder_factories.copy()


def get_embedder_factory(key: str) -> ImageEmbedderFactory | None:
    """Returns the embedder factory for the given key."""
    return _embedder_factories.get(key)
