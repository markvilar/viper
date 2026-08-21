"""
Forge subgroup of the viper CLI.

``commands`` owns the ``forge`` command surface (parsing, framework wiring);
``actions`` owns the work (resolve a forge, apply it, write a checkpoint) with no
CLI-framework details so it is unit-testable without a parser.
"""

from .commands import forge_group

__all__ = [
    "forge_group",
]
