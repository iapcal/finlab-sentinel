"""Abstract storage backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class BackupMetadata:
    """Metadata for a backup entry."""

    dataset: str
    backup_key: str
    content_hash: str
    created_at: datetime
    row_count: int
    column_count: int
    file_path: Path
    file_size_bytes: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "dataset": self.dataset,
            "backup_key": self.backup_key,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "file_path": str(self.file_path),
            "file_size_bytes": self.file_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BackupMetadata:
        """Create from dictionary."""
        return cls(
            dataset=data["dataset"],
            backup_key=data["backup_key"],
            content_hash=data["content_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            row_count=data["row_count"],
            column_count=data["column_count"],
            file_path=Path(data["file_path"]),
            file_size_bytes=data.get("file_size_bytes", 0),
        )


class StorageBackend(ABC):
    """Abstract base class for backup storage."""

    @abstractmethod
    def save(
        self,
        backup_key: str,
        dataset: str,
        data: pd.DataFrame,
        content_hash: str,
    ) -> BackupMetadata:
        """Save DataFrame to storage.

        Args:
            backup_key: Unique key for this backup
            dataset: Original dataset name
            data: DataFrame to save
            content_hash: Pre-computed content hash

        Returns:
            Metadata for the saved backup
        """
        ...

    @abstractmethod
    def load_latest(
        self, backup_key: str
    ) -> tuple[pd.DataFrame, BackupMetadata] | None:
        """Load most recent backup for key.

        Args:
            backup_key: The backup key to load

        Returns:
            Tuple of (DataFrame, Metadata) or None if not found
        """
        ...

    @abstractmethod
    def load_by_date(
        self,
        backup_key: str,
        date: datetime,
    ) -> tuple[pd.DataFrame, BackupMetadata] | None:
        """Load backup for specific date.

        Args:
            backup_key: The backup key
            date: Date to load backup from

        Returns:
            Tuple of (DataFrame, Metadata) or None if not found
        """
        ...

    @abstractmethod
    def load_at_time(
        self,
        backup_key: str,
        target_time: datetime,
    ) -> tuple[pd.DataFrame, BackupMetadata] | None:
        """Load backup at or before specific datetime.

        Args:
            backup_key: The backup key
            target_time: Target datetime to load backup from

        Returns:
            Tuple of (DataFrame, Metadata) or None if not found
        """
        ...

    @abstractmethod
    def get_latest_metadata(self, backup_key: str) -> BackupMetadata | None:
        """Get metadata for most recent backup without loading data.

        Args:
            backup_key: The backup key

        Returns:
            Metadata or None if not found
        """
        ...

    @abstractmethod
    def list_backups(
        self,
        backup_key: str | None = None,
    ) -> list[BackupMetadata]:
        """List all backups, optionally filtered by key.

        Args:
            backup_key: Optional filter by backup key

        Returns:
            List of backup metadata
        """
        ...

    @abstractmethod
    def cleanup_expired(self, retention_days: int) -> int:
        """Remove backups older than retention period.

        Args:
            retention_days: Number of days to retain

        Returns:
            Count of deleted backups
        """
        ...

    @abstractmethod
    def delete(
        self,
        backup_key: str,
        date: datetime | None = None,
    ) -> int:
        """Delete specific backup or all backups for key.

        Args:
            backup_key: The backup key
            date: Optional specific date to delete, or all if None

        Returns:
            Count of deleted backups
        """
        ...

    @abstractmethod
    def accept_new_data(
        self,
        backup_key: str,
        data: pd.DataFrame,
        content_hash: str,
        dataset: str,
        reason: str | None = None,
    ) -> BackupMetadata:
        """Accept new data as the baseline, marking previous as superseded.

        Args:
            backup_key: The backup key
            data: New DataFrame to accept
            content_hash: Content hash of new data
            dataset: Original dataset name
            reason: Optional reason for accepting

        Returns:
            Metadata for the new baseline
        """
        ...
