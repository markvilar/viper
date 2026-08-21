"""
Forge registry: maps a ``(model_key, method)`` pair to the forge that reshapes
that model with the given decomposition method.

This is the dispatch layer the CLI consumes to resolve a forge without a
hardcoded table. It parallels the embedder factory registry in
``viper.registry``: forges self-register here via the ``register_forge``
decorator, and the CLI looks them up by key.

Each entry also carries a ``label`` — the method component used when deriving a
checkpoint file name (e.g. ``svd`` selects the method, but the file is named
``...-svd-truncated-...`` after the underlying decomposition).
"""

from collections.abc import Callable
from dataclasses import dataclass

from viper.types import ImageEmbedder

type ForgeFn = Callable[..., ImageEmbedder]


@dataclass(frozen=True)
class ForgeEntry:
    """A registered forge: the callable plus its checkpoint-name label."""

    forge: ForgeFn
    label: str


_forges: dict[tuple[str, str], ForgeEntry] = dict()


def register_forge(
    model_key: str, method: str, label: str | None = None
) -> Callable[[ForgeFn], ForgeFn]:
    """
    Registers a forge for a ``(model_key, method)`` pair.

    Arguments:
        model_key - embedder registry key the forge applies to
        method    - decomposition method selector (e.g. "svd")
        label     - method component for derived checkpoint names; defaults to
                    ``method`` when omitted
    """

    def decorator(func: ForgeFn) -> ForgeFn:
        _forges[(model_key, method)] = ForgeEntry(forge=func, label=label or method)
        return func

    return decorator


def get_forge_registry() -> dict[tuple[str, str], ForgeEntry]:
    """Returns a copy of the forge registry."""
    return _forges.copy()


def get_forge(model_key: str, method: str) -> ForgeEntry | None:
    """Returns the forge entry for the given key, or None if unregistered."""
    return _forges.get((model_key, method))
