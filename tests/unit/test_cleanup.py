"""Tests for cleanup utilities."""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from finlab_sentinel.config.schema import SentinelConfig, StorageConfig
from finlab_sentinel.storage.cleanup import (
    calculate_directory_size,
    cleanup_on_startup,
    get_storage_info,
)
from finlab_sentinel.storage.parquet import ParquetStorage


@pytest.fixture
def config_with_storage(tmp_path: Path) -> SentinelConfig:
    """Create config with temp storage path."""
    return SentinelConfig(storage=StorageConfig(path=tmp_path, retention_days=7))


@pytest.fixture
def config_with_min_backups(tmp_path: Path) -> SentinelConfig:
    """Create config with custom min_backups_per_dataset."""
    return SentinelConfig(
        storage=StorageConfig(
            path=tmp_path, retention_days=7, min_backups_per_dataset=2
        )
    )


class TestCleanupOnStartup:
    """Tests for cleanup_on_startup function."""

    def test_no_expired_backups(
        self, config_with_storage: SentinelConfig, sample_df: pd.DataFrame
    ):
        """Verify no cleanup when no expired backups."""
        # Create a recent backup
        storage = ParquetStorage(
            base_path=config_with_storage.get_storage_path(),
            compression=config_with_storage.storage.compression,
        )
        storage.save("test", "test:dataset", sample_df, "hash123")

        # Should not delete anything
        deleted = cleanup_on_startup(config_with_storage)
        assert deleted == 0

    def test_returns_zero_for_empty_storage(self, config_with_storage: SentinelConfig):
        """Verify returns 0 when storage is empty."""
        deleted = cleanup_on_startup(config_with_storage)
        assert deleted == 0

    def test_keeps_min_backups_when_all_expired(
        self, config_with_min_backups: SentinelConfig, sample_df: pd.DataFrame
    ):
        """Verify cleanup keeps min_backups_per_dataset even if all expired."""
        storage = ParquetStorage(
            base_path=config_with_min_backups.get_storage_path(),
            compression=config_with_min_backups.storage.compression,
        )

        # Create 4 backups all older than retention_days (spaced by minutes for different filenames)
        base_time = datetime.now() - timedelta(days=30)
        for i in range(4):
            storage.save("test", "test:dataset", sample_df, f"hash{i}")
            # Manually set old timestamp (each 10 minutes apart)
            with storage.index._connect() as conn:
                old_date = (base_time - timedelta(minutes=i * 10)).isoformat()
                conn.execute(
                    "UPDATE backups SET created_at = ? WHERE content_hash = ?",
                    (old_date, f"hash{i}"),
                )

        # Cleanup should keep at least min_backups_per_dataset=2
        cleanup_on_startup(config_with_min_backups)

        # Should have 2 remaining in index
        remaining = storage.list_backups("test")
        assert len(remaining) == 2

    def test_respects_min_backups_config(self, tmp_path: Path, sample_df: pd.DataFrame):
        """Verify cleanup uses min_backups_per_dataset from config."""
        config = SentinelConfig(
            storage=StorageConfig(
                path=tmp_path,
                retention_days=7,
                min_backups_per_dataset=5,  # Keep at least 5
            )
        )
        storage = ParquetStorage(
            base_path=config.get_storage_path(),
            compression=config.storage.compression,
        )

        # Create 7 old backups (spaced by minutes for different filenames)
        base_time = datetime.now() - timedelta(days=30)
        for i in range(7):
            storage.save("test", "test:dataset", sample_df, f"hash{i}")
            with storage.index._connect() as conn:
                old_date = (base_time - timedelta(minutes=i * 10)).isoformat()
                conn.execute(
                    "UPDATE backups SET created_at = ? WHERE content_hash = ?",
                    (old_date, f"hash{i}"),
                )

        # Cleanup should keep at least min_backups_per_dataset=5
        cleanup_on_startup(config)

        # Should have 5 remaining in index
        remaining = storage.list_backups("test")
        assert len(remaining) == 5


class TestGetStorageInfo:
    """Tests for get_storage_info function."""

    def test_empty_storage(self, config_with_storage: SentinelConfig):
        """Verify stats for empty storage."""
        stats = get_storage_info(config_with_storage)

        assert stats["total_backups"] == 0
        assert stats["total_size_bytes"] == 0

    def test_with_backups(
        self, config_with_storage: SentinelConfig, sample_df: pd.DataFrame
    ):
        """Verify stats with existing backups."""
        storage = ParquetStorage(
            base_path=config_with_storage.get_storage_path(),
            compression=config_with_storage.storage.compression,
        )
        storage.save("test1", "test:dataset1", sample_df, "hash1")
        storage.save("test2", "test:dataset2", sample_df, "hash2")

        stats = get_storage_info(config_with_storage)

        assert stats["total_backups"] == 2
        assert stats["total_size_bytes"] > 0
        assert stats["unique_datasets"] == 2


class TestCalculateDirectorySize:
    """Tests for calculate_directory_size function."""

    def test_empty_directory(self, tmp_path: Path):
        """Verify size of empty directory is 0."""
        size = calculate_directory_size(tmp_path)
        assert size == 0

    def test_nonexistent_directory(self, tmp_path: Path):
        """Verify nonexistent directory returns 0."""
        nonexistent = tmp_path / "does_not_exist"
        size = calculate_directory_size(nonexistent)
        assert size == 0

    def test_directory_with_files(self, tmp_path: Path):
        """Verify correct size calculation."""
        # Create some files
        file1 = tmp_path / "file1.txt"
        file1.write_text("hello")  # 5 bytes

        file2 = tmp_path / "subdir" / "file2.txt"
        file2.parent.mkdir()
        file2.write_text("world!")  # 6 bytes

        size = calculate_directory_size(tmp_path)
        assert size == 11
