"""Tests for configuration."""

from pathlib import Path

import pytest

from finlab_sentinel.config.loader import load_config
from finlab_sentinel.config.schema import (
    AnomalyBehavior,
    PolicyMode,
    SentinelConfig,
    StorageConfig,
)
from finlab_sentinel.exceptions import ConfigurationError


class TestStorageConfig:
    """Tests for StorageConfig."""

    def test_default_values(self):
        """Verify default values are applied."""
        config = StorageConfig()

        assert config.retention_days == 7
        assert config.format == "parquet"
        assert config.compression == "zstd"

    def test_path_expansion(self):
        """Verify ~ is expanded in path."""
        config = StorageConfig(path="~/.finlab-sentinel")

        assert not str(config.path).startswith("~")

    def test_retention_days_validation(self):
        """Verify retention_days validation."""
        with pytest.raises(ValueError):
            StorageConfig(retention_days=0)

        with pytest.raises(ValueError):
            StorageConfig(retention_days=400)


class TestSentinelConfig:
    """Tests for SentinelConfig."""

    def test_default_values(self):
        """Verify default configuration."""
        config = SentinelConfig()

        assert config.comparison.rtol == 1e-5
        assert config.comparison.policies.default_mode == PolicyMode.APPEND_ONLY
        assert config.anomaly.behavior == AnomalyBehavior.RAISE

    def test_get_storage_path(self, tmp_path: Path):
        """Verify get_storage_path."""
        config = SentinelConfig(storage=StorageConfig(path=tmp_path))

        assert config.get_storage_path() == tmp_path

    def test_get_reports_path(self, tmp_path: Path):
        """Verify get_reports_path."""
        config = SentinelConfig(storage=StorageConfig(path=tmp_path))

        reports_path = config.get_reports_path()
        assert str(reports_path).startswith(str(tmp_path))

    def test_is_dataset_history_modifiable(self):
        """Verify blacklist check."""
        from finlab_sentinel.config.schema import (
            ComparisonConfig,
            ComparisonPoliciesConfig,
        )

        config = SentinelConfig(
            comparison=ComparisonConfig(
                policies=ComparisonPoliciesConfig(
                    history_modifiable=["fundamental:eps"]
                )
            )
        )

        assert config.is_dataset_history_modifiable("fundamental:eps")
        assert not config.is_dataset_history_modifiable("price:close")


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_defaults_when_no_file(self, tmp_path: Path, monkeypatch):
        """Verify defaults are used when no config file."""
        monkeypatch.chdir(tmp_path)

        config = load_config()

        assert config.storage.retention_days == 7

    def test_load_from_toml(self, tmp_path: Path):
        """Verify loading from TOML file."""
        config_content = """
[storage]
retention_days = 14
compression = "snappy"

[comparison]
rtol = 1e-6

[anomaly]
behavior = "warn_return_cached"
"""
        config_file = tmp_path / "sentinel.toml"
        config_file.write_text(config_content)

        config = load_config(config_file)

        assert config.storage.retention_days == 14
        assert config.storage.compression == "snappy"
        assert config.comparison.rtol == 1e-6
        assert config.anomaly.behavior == AnomalyBehavior.WARN_RETURN_CACHED

    def test_load_nonexistent_raises(self, tmp_path: Path):
        """Verify error when explicit file doesn't exist."""
        with pytest.raises(ConfigurationError):
            load_config(tmp_path / "nonexistent.toml")

    def test_load_invalid_toml_raises(self, tmp_path: Path):
        """Verify error on invalid TOML."""
        config_file = tmp_path / "invalid.toml"
        config_file.write_text("invalid [ toml content")

        with pytest.raises(ConfigurationError):
            load_config(config_file)

    def test_load_invalid_values_raises(self, tmp_path: Path):
        """Verify error on invalid config values."""
        config_content = """
[storage]
retention_days = -1
"""
        config_file = tmp_path / "invalid_values.toml"
        config_file.write_text(config_content)

        with pytest.raises(ConfigurationError):
            load_config(config_file)

    def test_env_var_override(self, tmp_path: Path, monkeypatch):
        """Verify FINLAB_SENTINEL_CONFIG env var."""
        config_content = """
[storage]
retention_days = 30
"""
        config_file = tmp_path / "env_config.toml"
        config_file.write_text(config_content)

        monkeypatch.setenv("FINLAB_SENTINEL_CONFIG", str(config_file))

        config = load_config()

        assert config.storage.retention_days == 30
