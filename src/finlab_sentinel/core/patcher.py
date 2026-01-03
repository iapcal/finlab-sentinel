"""Monkey patching logic for finlab data.get."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from finlab_sentinel.core.registry import (
    get_enabled,
    get_original,
    remove_original,
    set_enabled,
    store_original,
)
from finlab_sentinel.exceptions import (
    SentinelAlreadyEnabledError,
    SentinelNotEnabledError,
)

if TYPE_CHECKING:
    from finlab_sentinel.config.schema import SentinelConfig

logger = logging.getLogger(__name__)


def enable(config: SentinelConfig | None = None) -> None:
    """Enable sentinel monitoring by monkey patching finlab.data.get.

    Args:
        config: Optional configuration. If None, loads from sentinel.toml
                or uses defaults.

    Raises:
        ImportError: If finlab package is not installed.
        SentinelAlreadyEnabledError: If sentinel is already enabled.
    """
    if is_enabled():
        raise SentinelAlreadyEnabledError(
            "Sentinel is already enabled. Call disable() first."
        )

    # Load config if not provided
    if config is None:
        from finlab_sentinel.config.loader import load_config

        config = load_config()

    # Configure logging
    _configure_logging(config)

    # Import finlab
    try:
        from finlab import data
    except ImportError as e:
        raise ImportError(
            "finlab package is not installed. Install it with: pip install finlab"
        ) from e

    # Store original function
    store_original("data.get", data.get)

    # Create interceptor
    from finlab_sentinel.core.interceptor import DataInterceptor

    interceptor = DataInterceptor(data.get, config)

    # Monkey patch
    data.get = interceptor

    # Mark as enabled
    set_enabled(True)

    # Cleanup expired backups on startup
    from finlab_sentinel.storage.cleanup import cleanup_on_startup

    cleanup_on_startup(config)

    logger.info("finlab-sentinel enabled")


def disable() -> None:
    """Disable sentinel monitoring and restore original data.get.

    Raises:
        SentinelNotEnabledError: If sentinel is not currently enabled.
    """
    if not is_enabled():
        raise SentinelNotEnabledError("Sentinel is not currently enabled.")

    # Get original function
    original_get = get_original("data.get")

    if original_get is None:
        logger.warning("Original data.get not found in registry")
    else:
        # Restore original
        try:
            from finlab import data

            data.get = original_get
        except ImportError:
            pass  # finlab not available, nothing to restore

    # Clean up
    remove_original("data.get")
    set_enabled(False)

    logger.info("finlab-sentinel disabled")


def is_enabled() -> bool:
    """Check if sentinel monitoring is currently enabled.

    Returns:
        True if enabled, False otherwise
    """
    return get_enabled()


def _configure_logging(config: SentinelConfig) -> None:
    """Configure logging based on config.

    Args:
        config: Sentinel configuration
    """
    sentinel_logger = logging.getLogger("finlab_sentinel")

    # Set level
    level = getattr(logging, config.logging.level)
    sentinel_logger.setLevel(level)

    # Add file handler if configured
    if config.logging.file:
        file_path = config.logging.file
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        sentinel_logger.addHandler(file_handler)

    # Ensure there's at least a console handler if no handlers exist
    if not sentinel_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(
            logging.Formatter("%(levelname)s - %(name)s - %(message)s")
        )
        sentinel_logger.addHandler(console_handler)
