"""Tests for CLI commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from finlab_sentinel.cli.main import app
from finlab_sentinel.config.schema import SentinelConfig, StorageConfig
from finlab_sentinel.storage.parquet import ParquetStorage

runner = CliRunner()


@pytest.fixture
def mock_config(tmp_path: Path) -> SentinelConfig:
    """Create mock config for CLI tests."""
    return SentinelConfig(storage=StorageConfig(path=tmp_path))


@pytest.fixture
def populated_storage(tmp_path: Path, sample_df: pd.DataFrame) -> ParquetStorage:
    """Create storage with some test data."""
    storage = ParquetStorage(base_path=tmp_path / "backups")
    storage.save("test__dataset1", "test:dataset1", sample_df, "hash1")
    storage.save("test__dataset2", "test:dataset2", sample_df, "hash2")
    return storage


class TestListCommand:
    """Tests for list command."""

    def test_list_empty_storage(self, mock_config: SentinelConfig):
        """Verify list with no backups."""
        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "No backups found" in result.stdout

    def test_list_with_backups(
        self, mock_config: SentinelConfig, sample_df: pd.DataFrame
    ):
        """Verify list shows existing backups."""
        # Create some backups
        storage = ParquetStorage(
            base_path=mock_config.get_storage_path(),
            compression=mock_config.storage.compression,
        )
        storage.save("test__ds", "test:ds", sample_df, "hash123")

        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0
        assert "test:ds" in result.stdout

    def test_list_json_format(
        self, mock_config: SentinelConfig, sample_df: pd.DataFrame
    ):
        """Verify list with JSON format."""
        storage = ParquetStorage(
            base_path=mock_config.get_storage_path(),
            compression=mock_config.storage.compression,
        )
        storage.save("test__ds", "test:ds", sample_df, "hash123")

        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(app, ["list", "--format", "json"])

        assert result.exit_code == 0
        assert "[" in result.stdout  # JSON array

    def test_list_csv_format(
        self, mock_config: SentinelConfig, sample_df: pd.DataFrame
    ):
        """Verify list with CSV format."""
        storage = ParquetStorage(
            base_path=mock_config.get_storage_path(),
            compression=mock_config.storage.compression,
        )
        storage.save("test__ds", "test:ds", sample_df, "hash123")

        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(app, ["list", "--format", "csv"])

        assert result.exit_code == 0
        assert "dataset,backup_key" in result.stdout


class TestCleanupCommand:
    """Tests for cleanup command."""

    def test_cleanup_no_expired(self, mock_config: SentinelConfig):
        """Verify cleanup with no expired backups."""
        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(app, ["cleanup"])

        assert result.exit_code == 0
        assert "No backups older than" in result.stdout

    def test_cleanup_dry_run(
        self, mock_config: SentinelConfig, sample_df: pd.DataFrame
    ):
        """Verify cleanup dry run doesn't delete."""
        storage = ParquetStorage(
            base_path=mock_config.get_storage_path(),
            compression=mock_config.storage.compression,
        )
        storage.save("test__ds", "test:ds", sample_df, "hash123")

        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            # Use 0 days to make backup "expired"
            result = runner.invoke(app, ["cleanup", "--days", "0", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry run" in result.stdout

        # Backup should still exist
        assert storage.load_latest("test__ds") is not None


class TestExportCommand:
    """Tests for export command."""

    def test_export_not_found(self, mock_config: SentinelConfig):
        """Verify export error for missing dataset."""
        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(app, ["export", "missing:dataset"])

        assert result.exit_code == 1
        assert "No backup found" in result.stdout

    def test_export_to_parquet(
        self, mock_config: SentinelConfig, sample_df: pd.DataFrame, tmp_path: Path
    ):
        """Verify export to parquet."""
        storage = ParquetStorage(
            base_path=mock_config.get_storage_path(),
            compression=mock_config.storage.compression,
        )
        storage.save("test__ds", "test:ds", sample_df, "hash123")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(
                app, ["export", "test:ds", "--output", str(output_dir)]
            )

        assert result.exit_code == 0
        assert "Exported to" in result.stdout

    def test_export_to_csv(
        self, mock_config: SentinelConfig, sample_df: pd.DataFrame, tmp_path: Path
    ):
        """Verify export to CSV."""
        storage = ParquetStorage(
            base_path=mock_config.get_storage_path(),
            compression=mock_config.storage.compression,
        )
        storage.save("test__ds", "test:ds", sample_df, "hash123")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(
                app,
                ["export", "test:ds", "--output", str(output_dir), "--format", "csv"],
            )

        assert result.exit_code == 0
        # Check CSV file was created
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) == 1


class TestConfigCommand:
    """Tests for config command."""

    def test_config_show(self, mock_config: SentinelConfig):
        """Verify config show displays config."""
        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "show"])

        assert result.exit_code == 0
        assert "storage" in result.stdout

    def test_config_validate_success(self, mock_config: SentinelConfig):
        """Verify config validate passes for valid config."""
        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(app, ["config", "validate"])

        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_config_init(self, tmp_path: Path):
        """Verify config init creates file."""
        output_file = tmp_path / "new_sentinel.toml"

        result = runner.invoke(app, ["config", "init", "--output", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Created config file" in result.stdout

    def test_config_paths(self):
        """Verify config paths lists search paths."""
        result = runner.invoke(app, ["config", "paths"])

        assert result.exit_code == 0
        assert "sentinel.toml" in result.stdout

    def test_config_unknown_action(self):
        """Verify error for unknown action."""
        result = runner.invoke(app, ["config", "unknown"])

        assert result.exit_code == 1
        assert "Unknown action" in result.stdout


class TestInfoCommand:
    """Tests for info command."""

    def test_info_shows_stats(self, mock_config: SentinelConfig, sample_df: pd.DataFrame):
        """Verify info shows storage stats."""
        storage = ParquetStorage(
            base_path=mock_config.get_storage_path(),
            compression=mock_config.storage.compression,
        )
        storage.save("test__ds", "test:ds", sample_df, "hash123")

        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "Storage Path" in result.stdout
        assert "Total Backups" in result.stdout


class TestAcceptCommand:
    """Tests for accept command."""

    def test_accept_missing_dataset(self, mock_config: SentinelConfig):
        """Verify accept fails for missing dataset."""
        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            with patch(
                "finlab_sentinel.core.interceptor.accept_current_data", return_value=False
            ):
                result = runner.invoke(app, ["accept", "missing:dataset"])

        assert result.exit_code == 1
        assert "Failed to accept" in result.stdout

    def test_accept_success(self, mock_config: SentinelConfig):
        """Verify accept succeeds."""
        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            with patch(
                "finlab_sentinel.core.interceptor.accept_current_data", return_value=True
            ):
                result = runner.invoke(
                    app, ["accept", "test:dataset", "--reason", "Test reason"]
                )

        assert result.exit_code == 0
        assert "Accepted new data" in result.stdout


class TestDiffCommand:
    """Tests for diff command."""

    def test_diff_no_backup(self, mock_config: SentinelConfig):
        """Verify diff fails when no backup exists."""
        with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
            result = runner.invoke(app, ["diff", "missing:dataset"])

        assert result.exit_code == 1
        assert "No backup found" in result.stdout

    def test_diff_identical(
        self, mock_config: SentinelConfig, sample_df: pd.DataFrame
    ):
        """Verify diff shows no differences for identical data."""
        storage = ParquetStorage(
            base_path=mock_config.get_storage_path(),
            compression=mock_config.storage.compression,
        )
        storage.save("test__ds", "test:ds", sample_df, "hash123")

        # Mock finlab.data.get to return same data
        import sys
        from types import ModuleType

        mock_data = MagicMock()
        mock_data.get = MagicMock(return_value=sample_df.copy())

        mock_finlab = ModuleType("finlab")
        mock_finlab.data = mock_data

        sys.modules["finlab"] = mock_finlab

        try:
            with patch("finlab_sentinel.config.loader.load_config", return_value=mock_config):
                result = runner.invoke(app, ["diff", "test:ds"])

            assert result.exit_code == 0
            assert "No differences" in result.stdout
        finally:
            if "finlab" in sys.modules:
                del sys.modules["finlab"]


class TestFormatSize:
    """Tests for _format_size helper."""

    def test_bytes(self):
        """Verify bytes formatting."""
        from finlab_sentinel.cli.main import _format_size

        assert _format_size(500) == "500.0 B"

    def test_kilobytes(self):
        """Verify KB formatting."""
        from finlab_sentinel.cli.main import _format_size

        assert _format_size(1500) == "1.5 KB"

    def test_megabytes(self):
        """Verify MB formatting."""
        from finlab_sentinel.cli.main import _format_size

        assert _format_size(1500000) == "1.4 MB"

    def test_gigabytes(self):
        """Verify GB formatting."""
        from finlab_sentinel.cli.main import _format_size

        assert _format_size(1500000000) == "1.4 GB"

    def test_terabytes(self):
        """Verify TB formatting."""
        from finlab_sentinel.cli.main import _format_size

        assert _format_size(1500000000000) == "1.4 TB"
