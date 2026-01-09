"""SQLite-based backup metadata index."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from finlab_sentinel.storage.backend import BackupMetadata

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS backups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    backup_key TEXT NOT NULL,
    dataset TEXT NOT NULL,
    file_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    accepted_reason TEXT,
    UNIQUE(backup_key, created_at)
);

CREATE INDEX IF NOT EXISTS idx_backup_key ON backups(backup_key);
CREATE INDEX IF NOT EXISTS idx_created_at ON backups(created_at);
CREATE INDEX IF NOT EXISTS idx_content_hash ON backups(content_hash);
"""


class BackupIndex:
    """SQLite-based index for backup metadata."""

    def __init__(self, db_path: Path) -> None:
        """Initialize backup index.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure database and tables exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(CREATE_TABLE_SQL)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add(self, metadata: BackupMetadata, reason: str | None = None) -> None:
        """Add backup metadata to index.

        Args:
            metadata: Backup metadata to add
            reason: Optional reason (for accepted data)
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO backups
                (backup_key, dataset, file_path, content_hash, created_at,
                 row_count, column_count, file_size_bytes, accepted_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    metadata.backup_key,
                    metadata.dataset,
                    str(metadata.file_path),
                    metadata.content_hash,
                    metadata.created_at.isoformat(),
                    metadata.row_count,
                    metadata.column_count,
                    metadata.file_size_bytes,
                    reason,
                ),
            )
        logger.debug(f"Added backup to index: {metadata.backup_key}")

    def get_latest(self, backup_key: str) -> BackupMetadata | None:
        """Get most recent backup metadata for a key.

        Args:
            backup_key: The backup key to query

        Returns:
            Most recent metadata or None
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM backups
                WHERE backup_key = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (backup_key,),
            ).fetchone()

            if row is None:
                return None

            return self._row_to_metadata(row)

    def get_by_date(self, backup_key: str, date: datetime) -> BackupMetadata | None:
        """Get backup metadata for a specific date.

        Args:
            backup_key: The backup key
            date: Target date

        Returns:
            Metadata for that date or None
        """
        # Find backup closest to and not after the target date
        date_str = date.date().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM backups
                WHERE backup_key = ?
                AND date(created_at) <= ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (backup_key, date_str),
            ).fetchone()

            if row is None:
                return None

            return self._row_to_metadata(row)

    def get_at_time(
        self, backup_key: str, target_time: datetime
    ) -> BackupMetadata | None:
        """Get backup metadata at or before a specific datetime.

        Args:
            backup_key: The backup key
            target_time: Target datetime

        Returns:
            Most recent metadata at or before target_time, or None
        """
        # Find backup closest to and not after the target datetime
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM backups
                WHERE backup_key = ?
                AND created_at <= ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (backup_key, target_time.isoformat()),
            ).fetchone()

            if row is None:
                return None

            return self._row_to_metadata(row)

    def list_all(self, backup_key: str | None = None) -> list[BackupMetadata]:
        """List all backups, optionally filtered by key.

        Args:
            backup_key: Optional filter

        Returns:
            List of backup metadata
        """
        with self._connect() as conn:
            if backup_key:
                rows = conn.execute(
                    """
                    SELECT * FROM backups
                    WHERE backup_key = ?
                    ORDER BY created_at DESC
                    """,
                    (backup_key,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM backups ORDER BY created_at DESC"
                ).fetchall()

            return [self._row_to_metadata(row) for row in rows]

    def delete_expired(
        self, before_date: datetime, min_keep_per_key: int = 3
    ) -> list[BackupMetadata]:
        """Delete backups created before a date, keeping at least N per key.

        Args:
            before_date: Delete backups older than this
            min_keep_per_key: Minimum number of backups to keep per backup_key

        Returns:
            List of deleted backup metadata
        """
        with self._connect() as conn:
            # First, find IDs to keep (latest N per key)
            keep_ids: set[int] = set()
            unique_keys = [
                row["backup_key"]
                for row in conn.execute(
                    "SELECT DISTINCT backup_key FROM backups"
                ).fetchall()
            ]

            for key in unique_keys:
                rows = conn.execute(
                    """
                    SELECT id FROM backups
                    WHERE backup_key = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (key, min_keep_per_key),
                ).fetchall()
                keep_ids.update(row["id"] for row in rows)

            # Delete expired backups that are NOT in keep_ids
            if keep_ids:
                placeholders = ",".join("?" * len(keep_ids))
                rows = conn.execute(
                    f"""
                    SELECT * FROM backups
                    WHERE created_at < ? AND id NOT IN ({placeholders})
                    """,
                    (before_date.isoformat(), *keep_ids),
                ).fetchall()

                deleted = [self._row_to_metadata(row) for row in rows]

                conn.execute(
                    f"""
                    DELETE FROM backups
                    WHERE created_at < ? AND id NOT IN ({placeholders})
                    """,
                    (before_date.isoformat(), *keep_ids),
                )
            else:
                # No backups exist at all
                rows = conn.execute(
                    "SELECT * FROM backups WHERE created_at < ?",
                    (before_date.isoformat(),),
                ).fetchall()
                deleted = [self._row_to_metadata(row) for row in rows]
                conn.execute(
                    "DELETE FROM backups WHERE created_at < ?",
                    (before_date.isoformat(),),
                )

            logger.info(f"Deleted {len(deleted)} expired backups from index")
            return deleted

    def delete_by_key(
        self, backup_key: str, date: datetime | None = None
    ) -> list[BackupMetadata]:
        """Delete backups by key and optionally date.

        Args:
            backup_key: The backup key to delete
            date: Optional specific date

        Returns:
            List of deleted backup metadata
        """
        with self._connect() as conn:
            if date:
                rows = conn.execute(
                    """
                    SELECT * FROM backups
                    WHERE backup_key = ? AND date(created_at) = ?
                    """,
                    (backup_key, date.date().isoformat()),
                ).fetchall()
                conn.execute(
                    """
                    DELETE FROM backups
                    WHERE backup_key = ? AND date(created_at) = ?
                    """,
                    (backup_key, date.date().isoformat()),
                )
            else:
                rows = conn.execute(
                    "SELECT * FROM backups WHERE backup_key = ?",
                    (backup_key,),
                ).fetchall()
                conn.execute(
                    "DELETE FROM backups WHERE backup_key = ?",
                    (backup_key,),
                )

            return [self._row_to_metadata(row) for row in rows]

    def get_unique_keys(self) -> list[str]:
        """Get list of unique backup keys.

        Returns:
            List of unique backup keys
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT backup_key FROM backups ORDER BY backup_key"
            ).fetchall()
            return [row["backup_key"] for row in rows]

    def get_stats(self) -> dict:
        """Get storage statistics.

        Returns:
            Dictionary with stats
        """
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_backups,
                    COUNT(DISTINCT backup_key) as unique_datasets,
                    SUM(file_size_bytes) as total_size,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM backups
                """
            ).fetchone()

            return {
                "total_backups": row["total_backups"],
                "unique_datasets": row["unique_datasets"],
                "total_size_bytes": row["total_size"] or 0,
                "oldest": row["oldest"],
                "newest": row["newest"],
            }

    @staticmethod
    def _row_to_metadata(row: sqlite3.Row) -> BackupMetadata:
        """Convert SQLite row to BackupMetadata."""
        return BackupMetadata(
            dataset=row["dataset"],
            backup_key=row["backup_key"],
            content_hash=row["content_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            row_count=row["row_count"],
            column_count=row["column_count"],
            file_path=Path(row["file_path"]),
            file_size_bytes=row["file_size_bytes"],
        )
