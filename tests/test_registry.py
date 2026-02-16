"""
Tests for the image embedder factory registry, ensuring factories are registered,
retrievable, callable, and that the returned registry copy does not affect internal state.
"""

from typing import Any

from viper import (
    register_embedder_factory,
    get_embedder_factory_registry,
)
from viper import (
    ImageEmbedder,
    ImageEmbedderFactory,
)


def test_registers_factory_in_registry() -> None:
    # Arrange
    registry_before: dict[str, ImageEmbedderFactory] = get_embedder_factory_registry()
    assert "test_embedder" not in registry_before

    @register_embedder_factory("test_embedder")
    def factory() -> ImageEmbedder:
        class DummyEmbedder:
            def __call__(self, image: Any) -> list[float]:
                return [0.1, 0.2]

        return DummyEmbedder()  # type: ignore[return-value]

    # Act
    registry_after: dict[str, ImageEmbedderFactory] = get_embedder_factory_registry()

    # Assert
    assert "test_embedder" in registry_after
    assert registry_after["test_embedder"] is factory


def test_registered_factory_is_callable_and_returns_embedder() -> None:
    # Arrange
    @register_embedder_factory("callable_embedder")
    def factory() -> ImageEmbedder:
        class DummyEmbedder:
            def __call__(self, image: Any) -> list[float]:
                return [1.0, 2.0, 3.0]

        return DummyEmbedder()  # type: ignore[return-value]

    registry: dict[str, ImageEmbedderFactory] = get_embedder_factory_registry()
    retrieved_factory: ImageEmbedderFactory = registry["callable_embedder"]

    # Act
    embedder: ImageEmbedder = retrieved_factory()
    result: list[float] = embedder("dummy-image")

    # Assert
    assert isinstance(result, list)
    assert result == [1.0, 2.0, 3.0]


def test_registry_copy_is_isolated() -> None:
    # Arrange
    registry: dict[str, ImageEmbedderFactory] = get_embedder_factory_registry()

    # Act
    registry["new_key"] = lambda: None  # type: ignore[assignment]

    # Assert
    # Modifying the returned dict must not affect the internal registry
    internal_registry: dict[str, ImageEmbedderFactory] = get_embedder_factory_registry()
    assert "new_key" not in internal_registry
