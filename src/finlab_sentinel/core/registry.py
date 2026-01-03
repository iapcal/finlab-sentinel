"""Registry for storing original functions."""

from __future__ import annotations

from collections.abc import Callable

# Global registry for original functions
_original_functions: dict[str, Callable] = {}

# Global flag for enabled state
_enabled: bool = False


def store_original(name: str, func: Callable) -> None:
    """Store original function in registry.

    Args:
        name: Name/key for the function
        func: Original function to store
    """
    _original_functions[name] = func


def get_original(name: str) -> Callable | None:
    """Get original function from registry.

    Args:
        name: Name/key for the function

    Returns:
        Original function or None if not found
    """
    return _original_functions.get(name)


def remove_original(name: str) -> Callable | None:
    """Remove and return original function from registry.

    Args:
        name: Name/key for the function

    Returns:
        Removed function or None if not found
    """
    return _original_functions.pop(name, None)


def clear_registry() -> None:
    """Clear all stored functions."""
    _original_functions.clear()


def set_enabled(enabled: bool) -> None:
    """Set enabled state.

    Args:
        enabled: Whether sentinel is enabled
    """
    global _enabled
    _enabled = enabled


def get_enabled() -> bool:
    """Get enabled state.

    Returns:
        Whether sentinel is enabled
    """
    return _enabled
