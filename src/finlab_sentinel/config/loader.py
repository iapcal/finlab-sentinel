"""Configuration loading utilities."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from finlab_sentinel.config.schema import SentinelConfig
from finlab_sentinel.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Default config file locations (in order of priority)
CONFIG_SEARCH_PATHS = [
    Path("sentinel.toml"),  # Current directory
    Path("~/.config/finlab-sentinel/sentinel.toml"),
    Path("~/.finlab-sentinel/sentinel.toml"),
]


def _find_config_file() -> Path | None:
    """Find configuration file in default locations."""
    for path in CONFIG_SEARCH_PATHS:
        expanded = path.expanduser()
        if expanded.exists():
            return expanded
    return None


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML file."""
    try:
        # Python 3.11+ has tomllib in stdlib
        import tomllib

        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        # Fallback for older Python versions
        try:
            import tomli

            with open(path, "rb") as f:
                return tomli.load(f)
        except ImportError as e:
            raise ConfigurationError(
                "tomli package required for Python < 3.11. "
                "Install with: pip install tomli"
            ) from e


def _load_callback(callback_path: str) -> Callable:
    """Load callback function from module path string.

    Args:
        callback_path: String in format "module.path:function_name"

    Returns:
        The loaded callback function

    Raises:
        ConfigurationError: If callback cannot be loaded
    """
    try:
        module_path, func_name = callback_path.rsplit(":", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        if not callable(func):
            raise ConfigurationError(f"'{callback_path}' is not callable")
        return func
    except ValueError as e:
        raise ConfigurationError(
            f"Invalid callback format '{callback_path}'. "
            "Expected format: 'module.path:function_name'"
        ) from e
    except ImportError as e:
        raise ConfigurationError(
            f"Cannot import module for callback '{callback_path}': {e}"
        ) from e
    except AttributeError as e:
        raise ConfigurationError(
            f"Function not found in callback '{callback_path}': {e}"
        ) from e


def load_config(
    config_path: Path | None = None,
    *,
    auto_discover: bool = True,
) -> SentinelConfig:
    """Load sentinel configuration.

    Configuration is loaded with the following priority:
    1. Explicit config_path if provided
    2. FINLAB_SENTINEL_CONFIG environment variable
    3. Auto-discovered config files (if auto_discover=True)
    4. Default values

    Args:
        config_path: Explicit path to configuration file
        auto_discover: Whether to search for config files in default locations

    Returns:
        Loaded SentinelConfig

    Raises:
        ConfigurationError: If configuration file cannot be loaded or is invalid
    """
    config_data: dict[str, Any] = {}

    # Determine config file path
    effective_path: Path | None = None

    if config_path is not None:
        effective_path = config_path.expanduser()
        if not effective_path.exists():
            raise ConfigurationError(f"Config file not found: {effective_path}")
    elif env_path := os.environ.get("FINLAB_SENTINEL_CONFIG"):
        effective_path = Path(env_path).expanduser()
        if not effective_path.exists():
            raise ConfigurationError(
                f"Config file from FINLAB_SENTINEL_CONFIG not found: {effective_path}"
            )
    elif auto_discover:
        effective_path = _find_config_file()

    # Load config file if found
    if effective_path is not None:
        logger.info(f"Loading configuration from: {effective_path}")
        try:
            config_data = _load_toml(effective_path)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse config file {effective_path}: {e}"
            ) from e
    else:
        logger.debug("No configuration file found, using defaults")

    # Create config object
    try:
        config = SentinelConfig(**config_data)
    except Exception as e:
        raise ConfigurationError(f"Invalid configuration: {e}") from e

    # Load callback if specified
    if config.anomaly.callback:
        callback_fn = _load_callback(config.anomaly.callback)
        config.anomaly.set_callback(callback_fn)

    return config


def create_default_config_file(path: Path) -> None:
    """Create a default configuration file.

    Args:
        path: Path where to create the config file
    """
    default_content = """\
# finlab-sentinel configuration file
# See https://github.com/yourusername/finlab-sentinel for documentation

[storage]
# Base path for backup storage (supports ~ expansion)
path = "~/.finlab-sentinel/"

# Number of days to retain backups
retention_days = 7

# Parquet compression algorithm
compression = "zstd"

[comparison]
# Relative tolerance for numeric comparisons
rtol = 1e-5

# Absolute tolerance for numeric comparisons
atol = 1e-8

# Check for dtype changes
check_dtype = true

# Check for NA type differences (pd.NA vs np.nan vs None)
check_na_type = true

# Change threshold (ratio of cells changed)
change_threshold = 0.10

[comparison.policies]
# Default comparison mode: "append_only", "threshold", or "permissive"
default_mode = "append_only"

# Datasets that can modify historical data (use permissive mode)
history_modifiable = []

[anomaly]
# Behavior when anomaly detected: "raise", "warn_return_cached", "warn_return_new"
behavior = "raise"

# Save anomaly reports to files
save_reports = true

# Directory for anomaly reports (relative to storage.path)
reports_dir = "reports/"

# Optional callback for notifications (format: "module.path:function_name")
# callback = "myproject.notifications:send_alert"

[logging]
# Log level: DEBUG, INFO, WARNING, ERROR
level = "INFO"

# Optional log file path
# file = "~/.finlab-sentinel/sentinel.log"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_content)
    logger.info(f"Created default configuration file: {path}")
