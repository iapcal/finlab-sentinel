"""Tests for storage backend."""

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
