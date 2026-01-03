"""Tests for monkey patching logic."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from finlab_sentinel.config.schema import SentinelConfig, StorageConfig
from finlab_sentinel.exceptions import (
    SentinelAlreadyEnabledError,
    SentinelNotEnabledError,
)


@pytest.fixture
def mock_finlab():
    """Create a mock finlab module."""
    # Create mock module
    mock_data = MagicMock()
    mock_data.get = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3]}))

    mock_finlab = ModuleType("finlab")
    mock_finlab.data = mock_data

    # Add to sys.modules
    sys.modules["finlab"] = mock_finlab
    sys.modules["finlab.data"] = mock_data

    yield mock_finlab

    # Cleanup
    if "finlab" in sys.modules:
        del sys.modules["finlab"]
    if "finlab.data" in sys.modules:
        del sys.modules["finlab.data"]


@pytest.fixture
def config_for_patcher(tmp_path: Path) -> SentinelConfig:
    """Create config for patcher tests."""
    return SentinelConfig(storage=StorageConfig(path=tmp_path))


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry before each test."""
    from finlab_sentinel.core import registry

    registry._original_functions.clear()
    registry._enabled = False
    yield
    registry._original_functions.clear()
    registry._enabled = False


class TestEnable:
    """Tests for enable function."""

    def test_enable_patches_data_get(self, mock_finlab, config_for_patcher):
        """Verify enable patches finlab.data.get."""
        from finlab_sentinel.core.patcher import enable, is_enabled

        original_get = mock_finlab.data.get

        enable(config_for_patcher)

        # Should be enabled
        assert is_enabled()

        # data.get should be patched (not the original)
        assert mock_finlab.data.get is not original_get

    def test_enable_raises_if_already_enabled(self, mock_finlab, config_for_patcher):
        """Verify error when enabling twice."""
        from finlab_sentinel.core.patcher import enable

        enable(config_for_patcher)

        with pytest.raises(SentinelAlreadyEnabledError):
            enable(config_for_patcher)

    def test_enable_without_config_loads_default(self, mock_finlab, tmp_path):
        """Verify enable loads config from default locations."""
        from finlab_sentinel.core.patcher import enable, is_enabled

        # Create a config file in tmp_path
        # Use as_posix() to avoid Windows backslash escaping issues in TOML
        config_file = tmp_path / "sentinel.toml"
        config_file.write_text(f'[storage]\npath = "{tmp_path.as_posix()}"\n')

        # Patch the config loading to use our temp config
        with patch("finlab_sentinel.config.loader.CONFIG_SEARCH_PATHS", [config_file]):
            enable(None)

        assert is_enabled()

    def test_enable_raises_if_finlab_not_installed(self, config_for_patcher):
        """Verify error when finlab is not installed."""
        from finlab_sentinel.core.patcher import enable

        # Remove finlab from sys.modules if present
        if "finlab" in sys.modules:
            del sys.modules["finlab"]

        with pytest.raises(ImportError, match="finlab package is not installed"):
            enable(config_for_patcher)


class TestDisable:
    """Tests for disable function."""

    def test_disable_restores_original(self, mock_finlab, config_for_patcher):
        """Verify disable restores original function."""
        from finlab_sentinel.core.patcher import disable, enable, is_enabled

        original_get = mock_finlab.data.get

        enable(config_for_patcher)
        assert is_enabled()

        disable()

        assert not is_enabled()
        assert mock_finlab.data.get is original_get

    def test_disable_raises_if_not_enabled(self):
        """Verify error when disabling without enabling."""
        from finlab_sentinel.core.patcher import disable

        with pytest.raises(SentinelNotEnabledError):
            disable()


class TestIsEnabled:
    """Tests for is_enabled function."""

    def test_initially_disabled(self):
        """Verify initially returns False."""
        from finlab_sentinel.core.patcher import is_enabled

        assert not is_enabled()

    def test_returns_true_after_enable(self, mock_finlab, config_for_patcher):
        """Verify returns True after enable."""
        from finlab_sentinel.core.patcher import enable, is_enabled

        enable(config_for_patcher)
        assert is_enabled()

    def test_returns_false_after_disable(self, mock_finlab, config_for_patcher):
        """Verify returns False after disable."""
        from finlab_sentinel.core.patcher import disable, enable, is_enabled

        enable(config_for_patcher)
        disable()
        assert not is_enabled()


class TestConfigureLogging:
    """Tests for _configure_logging function."""

    def test_configures_log_level(self, config_for_patcher):
        """Verify log level is configured."""
        import logging

        from finlab_sentinel.core.patcher import _configure_logging

        config_for_patcher.logging.level = "DEBUG"
        _configure_logging(config_for_patcher)

        logger = logging.getLogger("finlab_sentinel")
        assert logger.level == logging.DEBUG

    def test_configures_file_handler(self, config_for_patcher, tmp_path):
        """Verify file handler is added when configured."""
        import logging

        from finlab_sentinel.config.schema import LoggingConfig
        from finlab_sentinel.core.patcher import _configure_logging

        log_file = tmp_path / "logs" / "sentinel.log"
        config_for_patcher.logging = LoggingConfig(file=log_file)

        _configure_logging(config_for_patcher)

        logger = logging.getLogger("finlab_sentinel")

        # Check for file handler
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) >= 1

        # Cleanup handlers
        for h in logger.handlers[:]:
            logger.removeHandler(h)
