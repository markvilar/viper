"""
Package for visual place recognition (VPR) models.
"""

from importlib.metadata import PackageNotFoundError, version as _version

import viper.models as models  # noqa: F401

from .registry import FactoryRegistry as FactoryRegistry
from .registry import register_embedder_factory as register_embedder_factory
from .registry import EmbedderRegistrationEntry as EmbedderRegistrationEntry
from .registry import register_embedder_entries as register_embedder_entries
from .registry import get_embedder_factory_registry as get_embedder_factory_registry
from .registry import get_embedder_factory as get_embedder_factory
from .registry import get_embedder_families as get_embedder_families

from .types import ImageEmbedder as ImageEmbedder
from .types import ImageEmbedderFactory as ImageEmbedderFactory

try:
    __version__ = _version("libviper")
except PackageNotFoundError:  # package is not installed
    __version__ = "unknown"

__all__ = ["__version__"]
