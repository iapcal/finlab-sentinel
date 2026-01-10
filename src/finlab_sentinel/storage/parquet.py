"""Parquet-based storage implementation."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from finlab_sentinel.storage.backend import BackupMetadata, StorageBackend
from finlab_sentinel.storage.index import BackupIndex

logger = logging.getLogger(__name__)


def sanitize_backup_key(dataset: str, universe_hash: str | None = None) -> str:
    """Convert dataset name to filesystem-safe backup key.

    Args:
        dataset: Original dataset name (e.g., "price:收盤價")
        universe_hash: Optional hash of universe settings

    Returns:
        Sanitized backup key
    """
    # Replace special characters
    safe_name = dataset.replace(":", "__").replace("/", "_").replace("\\", "_")
    # Remove any other problematic characters
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("_", "-", "."))

    if universe_hash:
        safe_name = f"{safe_name}__universe_{universe_hash}"

    return safe_name


def generate_universe_hash(universe: object) -> str:
    """Generate hash for universe settings.

    Args:
        universe: Universe configuration object

    Returns:
        8-character hash string
    """
    return hashlib.md5(str(universe).encode()).hexdigest()[:8]


class ParquetStorage(StorageBackend):
    """Parquet-based storage backend."""

    def __init__(
        self,
        base_path: Path,
        compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd",
    ) -> None:
        """Initialize Parquet storage.

        Args:
            base_path: Base directory for storage
            compression: Compression algorithm to use
        """
        self.base_path = base_path.expanduser()
        self.data_path = self.base_path / "data" / "backups"
        self.compression = compression if compression != "none" else None

        # Ensure directories exist
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Initialize index
        self.index = BackupIndex(self.base_path / "data" / "index.sqlite")

        logger.debug(f"Initialized ParquetStorage at {self.base_path}")

    def _get_backup_dir(self, backup_key: str) -> Path:
        """Get directory for a backup key."""
        return self.data_path / backup_key

    def _get_backup_file(self, backup_key: str, date: datetime) -> Path:
        """Get file path for a specific backup."""
        # Use second-level timestamp to avoid overwrites
        timestamp = date.strftime("%Y-%m-%dT%H-%M-%S")
        return self._get_backup_dir(backup_key) / f"{timestamp}.parquet"

    def save(
        self,
        backup_key: str,
        dataset: str,
        data: pd.DataFrame,
        content_hash: str,
    ) -> BackupMetadata:
        """Save DataFrame to Parquet storage."""
        now = datetime.now()
        backup_dir = self._get_backup_dir(backup_key)
        backup_dir.mkdir(parents=True, exist_ok=True)

        file_path = self._get_backup_file(backup_key, now)

        # Convert to PyArrow table with metadata
        table = pa.Table.from_pandas(data)
        metadata = {
            b"sentinel_version": b"0.1.7",
            b"created_at": now.isoformat().encode(),
            b"content_hash": content_hash.encode(),
            b"dataset": dataset.encode(),
            b"backup_key": backup_key.encode(),
        }
        table = table.replace_schema_metadata({**table.schema.metadata, **metadata})

        # Write to Parquet
        pq.write_table(table, file_path, compression=self.compression)

        file_size = file_path.stat().st_size

        # Create metadata
        backup_metadata = BackupMetadata(
            dataset=dataset,
            backup_key=backup_key,
            content_hash=content_hash,
            created_at=now,
            row_count=len(data),
            column_count=len(data.columns),
            file_path=file_path,
            file_size_bytes=file_size,
        )

        # Add to index
        self.index.add(backup_metadata)

        logger.info(
            f"Saved backup: {backup_key} ({len(data)} rows, "
            f"{len(data.columns)} columns, {file_size:,} bytes)"
        )

        return backup_metadata

    def load_latest(
        self, backup_key: str
    ) -> tuple[pd.DataFrame, BackupMetadata] | None:
        """Load most recent backup for key."""
        metadata = self.index.get_latest(backup_key)
        if metadata is None:
            return None

        return self._load_from_metadata(metadata)

    def load_by_date(
        self,
        backup_key: str,
        date: datetime,
    ) -> tuple[pd.DataFrame, BackupMetadata] | None:
        """Load backup for specific date."""
        metadata = self.index.get_by_date(backup_key, date)
        if metadata is None:
            return None

        return self._load_from_metadata(metadata)

    def load_at_time(
        self,
        backup_key: str,
        target_time: datetime,
    ) -> tuple[pd.DataFrame, BackupMetadata] | None:
        """Load backup at or before specific datetime."""
        metadata = self.index.get_at_time(backup_key, target_time)
        if metadata is None:
            return None

        return self._load_from_metadata(metadata)

    def _load_from_metadata(
        self, metadata: BackupMetadata
    ) -> tuple[pd.DataFrame, BackupMetadata] | None:
        """Load DataFrame from metadata."""
        if not metadata.file_path.exists():
            logger.warning(f"Backup file not found: {metadata.file_path}")
            return None

        table = pq.read_table(metadata.file_path)
        df = table.to_pandas()

        return df, metadata

    def get_latest_metadata(self, backup_key: str) -> BackupMetadata | None:
        """Get metadata for most recent backup without loading data."""
        return self.index.get_latest(backup_key)

    def list_backups(
        self,
        backup_key: str | None = None,
    ) -> list[BackupMetadata]:
        """List all backups, optionally filtered by key."""
        return self.index.list_all(backup_key)

    def cleanup_expired(self, retention_days: int, min_keep_per_key: int = 3) -> int:
        """Remove backups older than retention period.

        Args:
            retention_days: Delete backups older than this many days
            min_keep_per_key: Always keep at least this many backups per dataset

        Returns:
            Number of backups deleted
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted_metadata = self.index.delete_expired(cutoff, min_keep_per_key)

        # Delete actual files
        deleted_count = 0
        for metadata in deleted_metadata:
            if metadata.file_path.exists():
                try:
                    metadata.file_path.unlink()
                    deleted_count += 1

                    # Remove empty directories
                    parent = metadata.file_path.parent
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                except OSError as e:
                    logger.warning(f"Failed to delete {metadata.file_path}: {e}")

        logger.info(
            f"Cleaned up {deleted_count} expired backups "
            f"(retention: {retention_days} days)"
        )

        return deleted_count

    def delete(
        self,
        backup_key: str,
        date: datetime | None = None,
    ) -> int:
        """Delete specific backup or all backups for key."""
        deleted_metadata = self.index.delete_by_key(backup_key, date)

        # Delete actual files
        deleted_count = 0
        for metadata in deleted_metadata:
            if metadata.file_path.exists():
                try:
                    metadata.file_path.unlink()
                    deleted_count += 1
                except OSError as e:
                    logger.warning(f"Failed to delete {metadata.file_path}: {e}")

        # Remove empty directory
        backup_dir = self._get_backup_dir(backup_key)
        if backup_dir.exists() and not any(backup_dir.iterdir()):
            backup_dir.rmdir()

        return deleted_count

    def accept_new_data(
        self,
        backup_key: str,
        data: pd.DataFrame,
        content_hash: str,
        dataset: str,
        reason: str | None = None,
    ) -> BackupMetadata:
        """Accept new data as the baseline."""
        now = datetime.now()
        backup_dir = self._get_backup_dir(backup_key)
        backup_dir.mkdir(parents=True, exist_ok=True)

        file_path = self._get_backup_file(backup_key, now)

        # Convert to PyArrow table with metadata
        table = pa.Table.from_pandas(data)
        metadata_dict = {
            b"sentinel_version": b"0.1.7",
            b"created_at": now.isoformat().encode(),
            b"content_hash": content_hash.encode(),
            b"dataset": dataset.encode(),
            b"backup_key": backup_key.encode(),
            b"accepted": b"true",
        }
        if reason:
            metadata_dict[b"accepted_reason"] = reason.encode()

        table = table.replace_schema_metadata(
            {**table.schema.metadata, **metadata_dict}
        )

        # Write to Parquet
        pq.write_table(table, file_path, compression=self.compression)

        file_size = file_path.stat().st_size

        # Create metadata
        backup_metadata = BackupMetadata(
            dataset=dataset,
            backup_key=backup_key,
            content_hash=content_hash,
            created_at=now,
            row_count=len(data),
            column_count=len(data.columns),
            file_path=file_path,
            file_size_bytes=file_size,
        )

        # Add to index with reason
        self.index.add(backup_metadata, reason=reason)

        logger.info(
            f"Accepted new data as baseline: {backup_key}"
            + (f" (reason: {reason})" if reason else "")
        )

        return backup_metadata

    def get_stats(self) -> dict:
        """Get storage statistics."""
        stats = self.index.get_stats()
        stats["storage_path"] = str(self.base_path)
        return stats

    def get_unique_datasets(self) -> list[str]:
        """Get list of unique backup keys."""
        return self.index.get_unique_keys()
