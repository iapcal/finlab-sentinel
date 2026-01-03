"""Tests for configuration loading utilities."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from finlab_sentinel.config.loader import (
    _find_config_file,
    _load_callback,
    _load_toml,
    create_default_config_file,
    load_config,
)
from finlab_sentinel.config.schema import SentinelConfig
from finlab_sentinel.exceptions import ConfigurationError


class TestFindConfigFile:
    """Tests for _find_config_file function."""

    def test_returns_none_when_no_config(self):
        """Verify returns None when no config file exists."""
        with patch.object(Path, "exists", return_value=False):
            result = _find_config_file()
            assert result is None

    def test_returns_first_existing_config(self, tmp_path):
        """Verify returns first existing config file."""
        config_file = tmp_path / "sentinel.toml"
        config_file.write_text("[storage]\npath = './data'")

        with patch(
            "finlab_sentinel.config.loader.CONFIG_SEARCH_PATHS",
            [config_file],
        ):
            result = _find_config_file()
            assert result == config_file


class TestLoadToml:
    """Tests for _load_toml function."""

    def test_loads_valid_toml(self, tmp_path):
        """Verify valid TOML is loaded."""
        toml_file = tmp_path / "test.toml"
        toml_file.write_text('[storage]\npath = "~/.sentinel"')

        result = _load_toml(toml_file)

        assert result == {"storage": {"path": "~/.sentinel"}}

    def test_raises_on_invalid_toml(self, tmp_path):
        """Verify error on invalid TOML."""
        toml_file = tmp_path / "invalid.toml"
        toml_file.write_text("this is not valid [toml")

        with pytest.raises(Exception):
            _load_toml(toml_file)


class TestLoadCallback:
    """Tests for _load_callback function."""

    def test_loads_valid_callback(self):
        """Verify valid callback is loaded."""
        # Use a known function from stdlib
        callback = _load_callback("os.path:exists")
        assert callable(callback)

    def test_raises_on_invalid_format(self):
        """Verify error on invalid callback format."""
        with pytest.raises(ConfigurationError, match="Invalid callback format"):
            _load_callback("invalid_no_colon")

    def test_raises_on_missing_module(self):
        """Verify error on missing module."""
        with pytest.raises(ConfigurationError, match="Cannot import module"):
            _load_callback("nonexistent.module:func")

    def test_raises_on_missing_function(self):
        """Verify error on missing function."""
        with pytest.raises(ConfigurationError, match="Function not found"):
            _load_callback("os.path:nonexistent_func")

    def test_raises_on_non_callable(self):
        """Verify error when attribute is not callable."""
        with pytest.raises(ConfigurationError, match="is not callable"):
            _load_callback("os:name")  # os.name is a string, not callable


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_default_config(self):
        """Verify default config is loaded when no file exists."""
        config = load_config(auto_discover=False)

        assert isinstance(config, SentinelConfig)
        assert config.storage.retention_days == 7  # Default value

    def test_loads_from_explicit_path(self, tmp_path):
        """Verify config is loaded from explicit path."""
        config_file = tmp_path / "my_config.toml"
        config_file.write_text(
            """
[storage]
path = "/custom/path"
retention_days = 14
"""
        )

        config = load_config(config_file)

        assert config.storage.path == Path("/custom/path")
        assert config.storage.retention_days == 14

    def test_raises_on_missing_explicit_path(self, tmp_path):
        """Verify error when explicit config file is missing."""
        missing = tmp_path / "nonexistent.toml"

        with pytest.raises(ConfigurationError, match="Config file not found"):
            load_config(missing)

    def test_loads_from_env_variable(self, tmp_path):
        """Verify config is loaded from FINLAB_SENTINEL_CONFIG env var."""
        config_file = tmp_path / "env_config.toml"
        config_file.write_text("[storage]\nretention_days = 30")

        with patch.dict(os.environ, {"FINLAB_SENTINEL_CONFIG": str(config_file)}):
            config = load_config()

        assert config.storage.retention_days == 30

    def test_raises_on_missing_env_path(self, tmp_path):
        """Verify error when env var points to missing file."""
        missing = tmp_path / "missing.toml"

        with patch.dict(os.environ, {"FINLAB_SENTINEL_CONFIG": str(missing)}):
            with pytest.raises(
                ConfigurationError, match="FINLAB_SENTINEL_CONFIG not found"
            ):
                load_config()

    def test_raises_on_invalid_config_values(self, tmp_path):
        """Verify error on invalid config values."""
        config_file = tmp_path / "invalid.toml"
        config_file.write_text('[storage]\nretention_days = "not_a_number"')

        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            load_config(config_file)

    def test_auto_discover_false_skips_search(self, tmp_path):
        """Verify auto_discover=False skips file search."""
        # Create a config in current dir that would normally be found
        config_file = tmp_path / "sentinel.toml"
        config_file.write_text("[storage]\nretention_days = 99")

        with patch(
            "finlab_sentinel.config.loader.CONFIG_SEARCH_PATHS",
            [config_file],
        ):
            # With auto_discover=False, should use defaults
            config = load_config(auto_discover=False)
            assert config.storage.retention_days == 7  # Default, not 99


class TestCreateDefaultConfigFile:
    """Tests for create_default_config_file function."""

    def test_creates_config_file(self, tmp_path):
        """Verify config file is created."""
        config_file = tmp_path / "new_config.toml"

        create_default_config_file(config_file)

        assert config_file.exists()
        content = config_file.read_text()
        assert "[storage]" in content
        assert "[comparison]" in content
        assert "[anomaly]" in content

    def test_creates_parent_directories(self, tmp_path):
        """Verify parent directories are created."""
        config_file = tmp_path / "nested" / "dir" / "config.toml"

        create_default_config_file(config_file)

        assert config_file.exists()
        assert config_file.parent.exists()

    def test_overwrites_existing_file(self, tmp_path):
        """Verify existing file is overwritten."""
        config_file = tmp_path / "existing.toml"
        config_file.write_text("old content")

        create_default_config_file(config_file)

        content = config_file.read_text()
        assert "old content" not in content
        assert "[storage]" in content
