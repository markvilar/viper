"""
Package for visual place recognition (VPR) models.
"""

import viper.models as models  # noqa: F401

from .registry import FactoryRegistry as FactoryRegistry
from .registry import register_embedder_factory as register_embedder_factory
from .registry import get_embedder_factory_registry as get_embedder_factory_registry
from .registry import get_embedder_factory as get_embedder_factory

from .types import ImageEmbedder as ImageEmbedder
from .types import ImageEmbedderFactory as ImageEmbedderFactory

__all__ = []
