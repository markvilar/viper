"""
Shared ``rich`` console for the viper CLI.

Provides a single ``Console`` instance so all CLI output goes through ``rich``,
giving consistent formatting and a place to grow richer reporting later.
"""

from rich.console import Console

console = Console()
"""Shared console for standard CLI output."""

error_console = Console(stderr=True)
"""Shared console for error output, written to stderr."""
