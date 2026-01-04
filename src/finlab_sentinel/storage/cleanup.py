"""Cleanup utilities for expired backups."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from finlab_sentinel.config.schema import SentinelConfig

logger = logging.getLogger(__name__)


def cleanup_on_startup(config: SentinelConfig) -> int:
    """Cleanup expired backups on startup.

    This is called when sentinel is enabled to remove old backups.

    Args:
        config: Sentinel configuration

    Returns:
        Number of backups cleaned up
    """
    from finlab_sentinel.storage.parquet import ParquetStorage

    storage = ParquetStorage(
        base_path=config.get_storage_path(),
        compression=config.storage.compression,
    )

    deleted_count = storage.cleanup_expired(
        config.storage.retention_days,
        config.storage.min_backups_per_dataset,
    )

    if deleted_count > 0:
        logger.info(
            f"Startup cleanup: removed {deleted_count} expired backups "
            f"(retention: {config.storage.retention_days} days)"
        )

    return deleted_count


def get_storage_info(config: SentinelConfig) -> dict:
    """Get storage information and statistics.

    Args:
        config: Sentinel configuration

    Returns:
        Dictionary with storage stats
    """
    from finlab_sentinel.storage.parquet import ParquetStorage

    storage = ParquetStorage(
        base_path=config.get_storage_path(),
        compression=config.storage.compression,
    )

    return storage.get_stats()


def calculate_directory_size(path: Path) -> int:
    """Calculate total size of a directory recursively.

    Args:
        path: Directory path

    Returns:
        Total size in bytes
    """
    if not path.exists():
        return 0

    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size

    return total
