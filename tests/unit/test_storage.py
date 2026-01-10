"""Tests for storage backend."""

import time
from datetime import datetime, timedelta

import pandas as pd

from finlab_sentinel.storage.parquet import ParquetStorage, sanitize_backup_key


class TestSanitizeBackupKey:
    """Tests for backup key sanitization."""

    def test_basic_dataset_name(self):
        """Verify basic dataset name is sanitized."""
        key = sanitize_backup_key("price:收盤價")
        assert key == "price__收盤價"

    def test_with_universe_hash(self):
        """Verify universe hash is appended."""
        key = sanitize_backup_key("price:收盤價", "abc123")
        assert key == "price__收盤價__universe_abc123"

    def test_removes_slashes(self):
        """Verify slashes are removed."""
        key = sanitize_backup_key("path/to/data")
        assert "/" not in key


class TestParquetStorage:
    """Tests for ParquetStorage class."""

    def test_save_creates_file(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify save creates parquet file."""
        metadata = parquet_storage.save(
            backup_key="test_dataset",
            dataset="test:dataset",
            data=sample_df,
            content_hash="abc123",
        )

        assert metadata.file_path.exists()
        assert metadata.row_count == len(sample_df)
        assert metadata.column_count == len(sample_df.columns)

    def test_load_returns_saved_data(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify loaded DataFrame matches saved."""
        parquet_storage.save(
            backup_key="test_dataset",
            dataset="test:dataset",
            data=sample_df,
            content_hash="abc123",
        )

        result = parquet_storage.load_latest("test_dataset")

        assert result is not None
        loaded_df, metadata = result
        # Check values match (ignore index freq attribute which may not be preserved)
        pd.testing.assert_frame_equal(
            loaded_df.reset_index(drop=True),
            sample_df.reset_index(drop=True),
        )
        # Check index values match
        assert list(loaded_df.index) == list(sample_df.index)

    def test_load_preserves_dtypes(self, parquet_storage: ParquetStorage):
        """Verify dtypes are preserved through save/load."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            }
        )

        parquet_storage.save(
            backup_key="dtype_test",
            dataset="test",
            data=df,
            content_hash="abc",
        )

        result = parquet_storage.load_latest("dtype_test")
        loaded_df, _ = result

        assert loaded_df["int_col"].dtype == df["int_col"].dtype
        assert loaded_df["float_col"].dtype == df["float_col"].dtype

    def test_list_backups(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify all backups are listed."""
        # Save multiple backups
        parquet_storage.save("ds1", "dataset1", sample_df, "hash1")
        parquet_storage.save("ds2", "dataset2", sample_df, "hash2")

        backups = parquet_storage.list_backups()

        assert len(backups) >= 2

    def test_list_backups_filtered(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify backups can be filtered by key."""
        parquet_storage.save("ds1", "dataset1", sample_df, "hash1")
        parquet_storage.save("ds2", "dataset2", sample_df, "hash2")

        backups = parquet_storage.list_backups("ds1")

        assert len(backups) == 1
        assert backups[0].backup_key == "ds1"

    def test_load_nonexistent_returns_none(self, parquet_storage: ParquetStorage):
        """Verify loading nonexistent backup returns None."""
        result = parquet_storage.load_latest("nonexistent")
        assert result is None

    def test_get_latest_metadata(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify getting metadata without loading data."""
        parquet_storage.save("test", "test", sample_df, "hash1")

        metadata = parquet_storage.get_latest_metadata("test")

        assert metadata is not None
        assert metadata.content_hash == "hash1"

    def test_delete_backup(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify backup deletion."""
        parquet_storage.save("to_delete", "test", sample_df, "hash")

        deleted = parquet_storage.delete("to_delete")

        assert deleted == 1
        assert parquet_storage.load_latest("to_delete") is None

    def test_accept_new_data(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify accept_new_data saves with reason."""
        metadata = parquet_storage.accept_new_data(
            backup_key="accepted",
            data=sample_df,
            content_hash="new_hash",
            dataset="test",
            reason="Data correction confirmed",
        )

        assert metadata is not None
        assert metadata.content_hash == "new_hash"

    def test_get_stats(self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame):
        """Verify storage statistics."""
        parquet_storage.save("stat_test", "test", sample_df, "hash")

        stats = parquet_storage.get_stats()

        assert stats["total_backups"] >= 1
        assert stats["total_size_bytes"] > 0

    def test_backup_file_uses_second_timestamp(self, parquet_storage: ParquetStorage):
        """Verify backup filename uses second-level timestamp."""
        dt = datetime(2024, 1, 4, 9, 30, 45)
        path = parquet_storage._get_backup_file("test_key", dt)
        assert path.name == "2024-01-04T09-30-45.parquet"

    def test_backup_file_different_minutes(self, parquet_storage: ParquetStorage):
        """Verify different seconds produce different filenames."""
        dt1 = datetime(2024, 1, 4, 9, 30, 0)
        dt2 = datetime(2024, 1, 4, 9, 31, 0)
        path1 = parquet_storage._get_backup_file("test_key", dt1)
        path2 = parquet_storage._get_backup_file("test_key", dt2)
        assert path1.name != path2.name
        assert path1.name == "2024-01-04T09-30-00.parquet"
        assert path2.name == "2024-01-04T09-31-00.parquet"

    def test_multiple_backups_same_day(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify multiple backups on the same day are not overwritten."""
        # Save first backup
        parquet_storage.save("multi_test", "test", sample_df, "hash1")
        # Wait a bit to ensure different timestamp
        time.sleep(1.0)
        # Modify and save again
        modified_df = sample_df.copy()
        modified_df.iloc[0, 0] = 999.99
        parquet_storage.save("multi_test", "test", modified_df, "hash2")

        # Both backups should exist
        backups = parquet_storage.list_backups("multi_test")
        assert len(backups) == 2

    def test_cleanup_expired_keeps_min_per_key(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify cleanup keeps minimum backups per key even if all expired."""
        # Create 5 backups with different old timestamps (spaced by minutes)
        # This ensures different filenames
        base_time = datetime.now() - timedelta(days=30)
        for i in range(5):
            parquet_storage.save("cleanup_test", "test", sample_df, f"hash{i}")
            # Set different old timestamps (each 10 minutes apart) to get different filenames
            with parquet_storage.index._connect() as conn:
                old_date = (base_time - timedelta(minutes=i * 10)).isoformat()
                conn.execute(
                    "UPDATE backups SET created_at = ? WHERE content_hash = ?",
                    (old_date, f"hash{i}"),
                )

        # All 5 are older than 7 days, but should keep 3 (in index)
        parquet_storage.cleanup_expired(retention_days=7, min_keep_per_key=3)

        # Index should have 3 remaining entries
        remaining = parquet_storage.list_backups("cleanup_test")
        assert len(remaining) == 3

    def test_cleanup_expired_respects_retention_when_enough_recent(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify cleanup deletes old backups when enough recent ones exist."""
        # Create 3 old backups with different timestamps
        base_old_time = datetime.now() - timedelta(days=30)
        for i in range(3):
            parquet_storage.save("retention_test", "test", sample_df, f"old_hash{i}")
            with parquet_storage.index._connect() as conn:
                old_date = (base_old_time - timedelta(minutes=i * 10)).isoformat()
                conn.execute(
                    "UPDATE backups SET created_at = ? WHERE content_hash = ?",
                    (old_date, f"old_hash{i}"),
                )

        # Create 3 recent backups with different timestamps
        base_new_time = datetime.now()
        for i in range(3):
            parquet_storage.save("retention_test", "test", sample_df, f"new_hash{i}")
            with parquet_storage.index._connect() as conn:
                new_date = (base_new_time - timedelta(minutes=i * 10)).isoformat()
                conn.execute(
                    "UPDATE backups SET created_at = ? WHERE content_hash = ?",
                    (new_date, f"new_hash{i}"),
                )

        # Should delete the 3 old ones, keep the 3 recent ones
        parquet_storage.cleanup_expired(retention_days=7, min_keep_per_key=3)

        # Index should have 3 remaining entries (the recent ones)
        remaining = parquet_storage.list_backups("retention_test")
        assert len(remaining) == 3
        # All remaining should be recent (new_hash*)
        for backup in remaining:
            assert backup.content_hash.startswith("new_hash")

    def test_load_by_date(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify loading backup by specific date."""
        parquet_storage.save("date_test", "test", sample_df, "hash1")

        today = datetime.now()
        result = parquet_storage.load_by_date("date_test", today)

        assert result is not None
        loaded_df, metadata = result
        assert metadata.content_hash == "hash1"

    def test_load_by_date_not_found(self, parquet_storage: ParquetStorage):
        """Verify load_by_date returns None for nonexistent date."""
        result = parquet_storage.load_by_date("nonexistent", datetime.now())
        assert result is None

    def test_load_at_time(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify loading backup at specific datetime."""
        # Save first backup
        parquet_storage.save("time_test", "test", sample_df, "hash1")
        first_backup_time = datetime.now()

        # Wait a bit and save second backup
        time.sleep(1.0)
        modified_df = sample_df.copy()
        modified_df.iloc[0, 0] = 999.99
        parquet_storage.save("time_test", "test", modified_df, "hash2")

        # Load at time between the two backups
        result = parquet_storage.load_at_time(
            "time_test", first_backup_time + timedelta(seconds=0.5)
        )

        assert result is not None
        loaded_df, metadata = result
        assert metadata.content_hash == "hash1"
        # Verify it's the first backup data
        assert loaded_df.iloc[0, 0] != 999.99

    def test_load_at_time_returns_latest_before_target(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify load_at_time returns most recent backup before target time."""
        base_time = datetime.now() - timedelta(hours=3)

        # Save 3 backups at different times
        for i in range(3):
            parquet_storage.save("multi_time_test", "test", sample_df, f"hash{i}")
            with parquet_storage.index._connect() as conn:
                backup_time = (base_time + timedelta(hours=i)).isoformat()
                conn.execute(
                    "UPDATE backups SET created_at = ? WHERE content_hash = ?",
                    (backup_time, f"hash{i}"),
                )

        # Query at time between backup 1 and 2
        target_time = base_time + timedelta(hours=1, minutes=30)
        result = parquet_storage.load_at_time("multi_time_test", target_time)

        assert result is not None
        _, metadata = result
        assert metadata.content_hash == "hash1"

    def test_load_at_time_not_found(self, parquet_storage: ParquetStorage):
        """Verify load_at_time returns None for nonexistent backup."""
        result = parquet_storage.load_at_time("nonexistent", datetime.now())
        assert result is None

    def test_load_at_time_before_all_backups(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify load_at_time returns None when target is before all backups."""
        parquet_storage.save("before_test", "test", sample_df, "hash1")

        # Query at time before any backups
        past_time = datetime.now() - timedelta(days=1)
        result = parquet_storage.load_at_time("before_test", past_time)

        assert result is None

    def test_load_missing_file_returns_none(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify loading returns None when file is missing but index exists."""
        import os

        parquet_storage.save("file_test", "test", sample_df, "hash1")

        # Delete the actual file but keep index entry
        metadata = parquet_storage.get_latest_metadata("file_test")
        if metadata and metadata.file_path.exists():
            os.remove(metadata.file_path)

        result = parquet_storage.load_latest("file_test")
        assert result is None

    def test_delete_with_date(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify deleting backup by specific date."""
        parquet_storage.save("delete_date_test", "test", sample_df, "hash1")
        time.sleep(1.0)
        parquet_storage.save("delete_date_test", "test", sample_df, "hash2")

        today = datetime.now()
        deleted = parquet_storage.delete("delete_date_test", today)

        # Should have deleted at least one
        assert deleted >= 1

    def test_delete_cleans_empty_directory(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify delete removes empty directory."""
        parquet_storage.save("empty_dir_test", "test", sample_df, "hash1")

        backup_dir = parquet_storage._get_backup_dir("empty_dir_test")
        assert backup_dir.exists()

        parquet_storage.delete("empty_dir_test")

        # Directory should be removed
        assert not backup_dir.exists()

    def test_get_unique_datasets(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify getting unique dataset list."""
        parquet_storage.save("unique1", "dataset1", sample_df, "hash1")
        parquet_storage.save("unique2", "dataset2", sample_df, "hash2")

        datasets = parquet_storage.get_unique_datasets()

        assert "unique1" in datasets
        assert "unique2" in datasets

    def test_accept_new_data_without_reason(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify accept_new_data works without reason."""
        metadata = parquet_storage.accept_new_data(
            backup_key="accepted_no_reason",
            data=sample_df,
            content_hash="new_hash",
            dataset="test",
            reason=None,
        )

        assert metadata is not None


class TestGenerateUniverseHash:
    """Tests for generate_universe_hash function."""

    def test_generate_universe_hash(self):
        """Verify universe hash generation."""
        from finlab_sentinel.storage.parquet import generate_universe_hash

        hash1 = generate_universe_hash("SP500")
        hash2 = generate_universe_hash("SP500")
        hash3 = generate_universe_hash("NASDAQ100")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 8  # 8-character hash


class TestBackupMetadata:
    """Tests for BackupMetadata dataclass."""

    def test_to_dict(self):
        """Verify to_dict method."""
        from pathlib import Path

        from finlab_sentinel.storage.backend import BackupMetadata

        metadata = BackupMetadata(
            dataset="test:dataset",
            backup_key="test__dataset",
            content_hash="abc123",
            created_at=datetime(2025, 1, 4, 10, 30),
            row_count=100,
            column_count=5,
            file_path=Path("/tmp/test.parquet"),
            file_size_bytes=1024,
        )

        result = metadata.to_dict()

        assert result["dataset"] == "test:dataset"
        assert result["backup_key"] == "test__dataset"
        assert result["content_hash"] == "abc123"
        assert result["row_count"] == 100
        assert result["column_count"] == 5
        assert result["file_size_bytes"] == 1024

    def test_from_dict(self):
        """Verify from_dict method."""
        from finlab_sentinel.storage.backend import BackupMetadata

        data = {
            "dataset": "test:dataset",
            "backup_key": "test__dataset",
            "content_hash": "abc123",
            "created_at": "2025-01-04T10:30:00",
            "row_count": 100,
            "column_count": 5,
            "file_path": "/tmp/test.parquet",
            "file_size_bytes": 1024,
        }

        metadata = BackupMetadata.from_dict(data)

        assert metadata.dataset == "test:dataset"
        assert metadata.backup_key == "test__dataset"
        assert metadata.content_hash == "abc123"
        assert metadata.row_count == 100
        assert metadata.column_count == 5


class TestBackupIndex:
    """Tests for BackupIndex class."""

    def test_get_unique_keys(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify getting unique backup keys."""
        parquet_storage.save("key1", "dataset1", sample_df, "hash1")
        parquet_storage.save("key2", "dataset2", sample_df, "hash2")
        parquet_storage.save("key1", "dataset1", sample_df, "hash3")

        keys = parquet_storage.index.get_unique_keys()

        assert "key1" in keys
        assert "key2" in keys
        assert len(keys) == 2

    def test_get_by_date(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify get_by_date returns correct backup."""
        parquet_storage.save("bydate_test", "test", sample_df, "hash1")

        today = datetime.now()
        metadata = parquet_storage.index.get_by_date("bydate_test", today)

        assert metadata is not None
        assert metadata.content_hash == "hash1"

    def test_get_by_date_not_found(self, parquet_storage: ParquetStorage):
        """Verify get_by_date returns None for missing date."""
        metadata = parquet_storage.index.get_by_date("nonexistent", datetime.now())
        assert metadata is None

    def test_get_at_time(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify get_at_time returns correct backup metadata."""
        parquet_storage.save("attime_test", "test", sample_df, "hash1")

        now = datetime.now()
        metadata = parquet_storage.index.get_at_time("attime_test", now)

        assert metadata is not None
        assert metadata.content_hash == "hash1"

    def test_get_at_time_not_found(self, parquet_storage: ParquetStorage):
        """Verify get_at_time returns None for missing backup."""
        metadata = parquet_storage.index.get_at_time("nonexistent", datetime.now())
        assert metadata is None

    def test_get_at_time_before_all_backups(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify get_at_time returns None when target is before all backups."""
        parquet_storage.save("before_all_test", "test", sample_df, "hash1")

        past_time = datetime.now() - timedelta(days=1)
        metadata = parquet_storage.index.get_at_time("before_all_test", past_time)

        assert metadata is None

    def test_delete_by_key_with_date(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify deleting by key and date."""
        parquet_storage.save("delkey_test", "test", sample_df, "hash1")

        today = datetime.now()
        deleted = parquet_storage.index.delete_by_key("delkey_test", today)

        assert len(deleted) >= 1

    def test_delete_by_key_all(
        self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame
    ):
        """Verify deleting all backups for a key."""
        parquet_storage.save("delall_test", "test", sample_df, "hash1")
        time.sleep(1.0)
        parquet_storage.save("delall_test", "test", sample_df, "hash2")

        deleted = parquet_storage.index.delete_by_key("delall_test", None)

        assert len(deleted) == 2

    def test_get_stats(self, parquet_storage: ParquetStorage, sample_df: pd.DataFrame):
        """Verify getting storage stats."""
        parquet_storage.save("stats_test", "test", sample_df, "hash1")

        stats = parquet_storage.index.get_stats()

        assert stats["total_backups"] >= 1
        assert stats["unique_datasets"] >= 1
        assert stats["total_size_bytes"] >= 0
