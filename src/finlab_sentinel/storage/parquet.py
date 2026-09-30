"""Parquet-based storage implementation."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from finlab_sentinel.exceptions import StorageError
from finlab_sentinel.storage.backend import BackupMetadata, StorageBackend
from finlab_sentinel.storage.index import BackupIndex

logger = logging.getLogger(__name__)


def get_index_path(base_path: Path) -> Path:
    """Get the backup index database path for a storage base directory.

    Args:
        base_path: Sentinel storage base directory

    Returns:
        Path of the SQLite index (may not exist yet)
    """
    return base_path.expanduser() / "data" / "index.sqlite"


def _fsync_file(path: Path) -> None:
    """Flush a written file (and, where supported, its directory) to disk."""
    # Opened for update: Windows can only flush handles with write access
    with open(path, "rb+") as f:
        os.fsync(f.fileno())
    # Directory fsync makes the new entry durable; unsupported on Windows.
    with contextlib.suppress(OSError):
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


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
        self.index = BackupIndex(get_index_path(self.base_path))

        logger.debug(f"Initialized ParquetStorage at {self.base_path}")

    def _get_backup_dir(self, backup_key: str) -> Path:
        """Get directory for a backup key."""
        return self.data_path / backup_key

    def _get_backup_file(self, backup_key: str, date: datetime) -> Path:
        """Get file path for a specific backup."""
        # Use second-level timestamp to avoid overwrites
        timestamp = date.strftime("%Y-%m-%dT%H-%M-%S")
        return self._get_backup_dir(backup_key) / f"{timestamp}.parquet"

    def _new_backup_file(self, backup_key: str, date: datetime) -> Path:
        """Get a path for a new backup file that does not exist yet.

        Several backups of one key can be written within the same second
        (e.g. a baseline that is saved and then immediately replaced).
        Reusing the second-level name would overwrite a file an older index
        entry still points to, and retention cleanup of that entry would
        later delete the newer baseline's data, so fall back to a
        microsecond-precision name.
        """
        file_path = self._get_backup_file(backup_key, date)
        if not file_path.exists():
            return file_path

        stem = date.strftime("%Y-%m-%dT%H-%M-%S-%f")
        file_path = file_path.with_name(f"{stem}.parquet")
        counter = 1
        while file_path.exists():
            counter += 1
            file_path = file_path.with_name(f"{stem}_{counter}.parquet")
        return file_path

    def _write_backup(
        self,
        backup_key: str,
        dataset: str,
        data: pd.DataFrame,
        content_hash: str,
        created_at: datetime,
        extra_metadata: dict[bytes, bytes] | None = None,
    ) -> BackupMetadata:
        """Write a backup file without adding it to the index.

        Args:
            backup_key: The backup key
            dataset: Original dataset name
            data: DataFrame to write
            content_hash: Content hash stored with the backup
            created_at: Backup timestamp (also used for the file name)
            extra_metadata: Additional parquet schema metadata

        Returns:
            Metadata describing the written file
        """
        backup_dir = self._get_backup_dir(backup_key)
        backup_dir.mkdir(parents=True, exist_ok=True)

        file_path = self._new_backup_file(backup_key, created_at)

        # Convert to PyArrow table with metadata
        table = pa.Table.from_pandas(data)
        metadata = {
            b"sentinel_version": b"0.1.9",
            b"created_at": created_at.isoformat().encode(),
            b"content_hash": content_hash.encode(),
            b"dataset": dataset.encode(),
            b"backup_key": backup_key.encode(),
            **(extra_metadata or {}),
        }
        table = table.replace_schema_metadata({**table.schema.metadata, **metadata})

        # Write to Parquet; never leave a partial file behind
        try:
            pq.write_table(table, file_path, compression=self.compression)
        except Exception:
            file_path.unlink(missing_ok=True)
            raise

        return BackupMetadata(
            dataset=dataset,
            backup_key=backup_key,
            content_hash=content_hash,
            created_at=created_at,
            row_count=len(data),
            column_count=len(data.columns),
            file_path=file_path,
            file_size_bytes=file_path.stat().st_size,
        )

    def save(
        self,
        backup_key: str,
        dataset: str,
        data: pd.DataFrame,
        content_hash: str,
    ) -> BackupMetadata:
        """Save DataFrame to Parquet storage."""
        backup_metadata = self._write_backup(
            backup_key, dataset, data, content_hash, datetime.now()
        )

        # Add to index
        self.index.add(backup_metadata)

        logger.info(
            f"Saved backup: {backup_key} ({len(data)} rows, "
            f"{len(data.columns)} columns, {backup_metadata.file_size_bytes:,} bytes)"
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

    def load_backup(self, metadata: BackupMetadata) -> pd.DataFrame | None:
        """Load the data of a specific backup.

        Args:
            metadata: Metadata of the backup (e.g. from get_latest_metadata)

        Returns:
            The DataFrame, or None if the backup file is missing
        """
        result = self._load_from_metadata(metadata)
        return result[0] if result is not None else None

    def get_latest_metadata(self, backup_key: str) -> BackupMetadata | None:
        """Get metadata for most recent backup without loading data."""
        return self.index.get_latest(backup_key)

    def list_backups(
        self,
        backup_key: str | None = None,
    ) -> list[BackupMetadata]:
        """List all backups, optionally filtered by key."""
        return self.index.list_all(backup_key)

    def _keep_file(self, metadata: BackupMetadata, still_used: set[Path]) -> bool:
        """Check whether a removed index entry's file must be kept.

        Versions up to 0.1.9 could point several index entries at one file
        (backups written within the same second), so the file of a removed
        entry may still hold the data of a newer, retained entry. Index
        entries store absolute paths, so a copied storage directory still
        points at the original's files; those are never deleted either.
        """
        if metadata.file_path in still_used:
            logger.warning(
                f"Keeping {metadata.file_path}: still used by another backup "
                f"of {metadata.backup_key}"
            )
            return True
        try:
            inside = metadata.file_path.resolve().is_relative_to(
                self.data_path.resolve()
            )
        except (OSError, ValueError):
            inside = False
        if not inside:
            logger.warning(
                f"Keeping {metadata.file_path}: outside this storage ({self.data_path})"
            )
            return True
        return False

    def cleanup_expired(self, retention_days: int, min_keep_per_key: int = 3) -> int:
        """Remove backups older than retention period.

        Args:
            retention_days: Delete backups older than this many days
            min_keep_per_key: Always keep at least this many backups per dataset

        Returns:
            Number of backups deleted
        """
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted_metadata = self.index.delete_expired(cutoff, min_keep_per_key)

        # Delete actual files
        still_used = self.index.referenced_files()
        deleted_count = 0
        for metadata in deleted_metadata:
            if self._keep_file(metadata, still_used):
                continue
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
        still_used = self.index.referenced_files()
        deleted_count = 0
        for metadata in deleted_metadata:
            if self._keep_file(metadata, still_used):
                continue
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
        extra_metadata = {b"accepted": b"true"}
        if reason:
            extra_metadata[b"accepted_reason"] = reason.encode()

        backup_metadata = self._write_backup(
            backup_key,
            dataset,
            data,
            content_hash,
            datetime.now(),
            extra_metadata,
        )

        # Add to index with reason
        self.index.add(backup_metadata, reason=reason)

        logger.info(
            f"Accepted new data as baseline: {backup_key}"
            + (f" (reason: {reason})" if reason else "")
        )

        return backup_metadata

    def restore_baseline(
        self,
        backup_key: str,
        dataset: str,
        data: pd.DataFrame,
        content_hash: str,
        patch_id: str,
        expected_latest: BackupMetadata | None,
        reason: str | None = None,
    ) -> BackupMetadata:
        """Make data restored from a permanent patch the latest baseline.

        ``content_hash`` is stored verbatim, never recomputed: it is the hash
        the baseline had when the patch was created, which DataInterceptor
        computes over preprocessed data.

        The new file is fully written and flushed to disk before the index
        entry that makes it the baseline is added, and the entry is only added
        if the latest backup is still ``expected_latest``. On any failure the
        new file is removed and the previous baseline stays in place.

        Args:
            backup_key: The backup key
            dataset: Original dataset name
            data: Baseline DataFrame preserved by the patch
            content_hash: Content hash recorded in the patch
            patch_id: The patch being restored (recorded in file metadata)
            expected_latest: Latest backup the restore was planned against
                (None if the key had no backups)
            reason: Optional reason stored with the index entry

        Returns:
            Metadata for the new baseline

        Raises:
            StorageError: If the latest backup changed since
                ``expected_latest`` was read
        """
        created_at = datetime.now()
        if expected_latest is not None and expected_latest.created_at >= created_at:
            # Clock went backwards or the latest backup is future-dated; the
            # restored baseline must still sort after it to become the latest.
            created_at = expected_latest.created_at + timedelta(microseconds=1)

        extra_metadata = {b"restored_from_patch": patch_id.encode()}
        if reason:
            extra_metadata[b"restored_reason"] = reason.encode()

        backup_metadata = self._write_backup(
            backup_key, dataset, data, content_hash, created_at, extra_metadata
        )

        try:
            _fsync_file(backup_metadata.file_path)
            added = self.index.add_if_latest(
                backup_metadata, expected_latest, reason=reason
            )
        except Exception:
            backup_metadata.file_path.unlink(missing_ok=True)
            raise

        if not added:
            backup_metadata.file_path.unlink(missing_ok=True)
            raise StorageError(f"Latest backup of {backup_key} changed during restore")

        logger.info(
            f"Restored baseline from patch {patch_id}: {backup_key}"
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
