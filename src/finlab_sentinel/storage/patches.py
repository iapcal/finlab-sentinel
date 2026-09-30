"""Permanent patch storage for accepted data changes.

When a user accepts anomalous data as the new baseline, a permanent patch
is created preserving the old baseline snapshot and a diff summary. Patches
live in their own directory and are never touched by retention cleanup.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq

from finlab_sentinel.comparison.differ import ComparisonResult
from finlab_sentinel.exceptions import PatchNotFoundError

logger = logging.getLogger(__name__)

PATCH_JSON_NAME = "patch.json"
OLD_DATA_NAME = "old_data.parquet"


@dataclass
class PatchMetadata:
    """Metadata for a permanent patch (mirrors patch.json)."""

    patch_id: str
    dataset: str
    backup_key: str
    created_at: datetime
    reason: str | None
    old_hash: str
    new_hash: str
    old_shape: tuple[int, int]
    new_shape: tuple[int, int]
    diff_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "patch_id": self.patch_id,
            "dataset": self.dataset,
            "backup_key": self.backup_key,
            "created_at": self.created_at.isoformat(),
            "reason": self.reason,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "old_shape": list(self.old_shape),
            "new_shape": list(self.new_shape),
            "diff_summary": self.diff_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PatchMetadata:
        """Create from dictionary."""
        return cls(
            patch_id=data["patch_id"],
            dataset=data["dataset"],
            backup_key=data["backup_key"],
            created_at=datetime.fromisoformat(data["created_at"]),
            reason=data.get("reason"),
            old_hash=data["old_hash"],
            new_hash=data["new_hash"],
            old_shape=tuple(data["old_shape"]),
            new_shape=tuple(data["new_shape"]),
            diff_summary=data.get("diff_summary", {}),
        )


def _build_diff_summary(result: ComparisonResult) -> dict:
    """Extract serializable diff summary from a comparison result."""
    return {
        "summary_text": result.summary(),
        "change_ratio": result.change_ratio,
        "added_rows": len(result.added_rows),
        "deleted_rows": len(result.deleted_rows),
        "added_columns": len(result.added_columns),
        "deleted_columns": len(result.deleted_columns),
        "modified_cells": max(result.modified_cells_count, len(result.modified_cells)),
        "na_type_changes": max(
            result.na_type_changes_count, len(result.na_type_changes)
        ),
        "dtype_changes": len(result.dtype_changes),
    }


class PatchStore:
    """Manages permanent patches under <base_path>/patches/."""

    def __init__(
        self,
        base_path: Path,
        compression: Literal["zstd", "snappy", "gzip", "none"] = "zstd",
    ) -> None:
        """Initialize patch store.

        Args:
            base_path: Sentinel storage base directory
            compression: Compression algorithm for old data snapshots
        """
        self.base_path = base_path.expanduser()
        self.patches_path = self.base_path / "patches"
        self.compression = compression if compression != "none" else None

    def _get_patch_dir(self, patch_id: str) -> Path:
        # Patch ids are single path components; reject anything that could
        # resolve outside the patches directory (e.g. "..", "../data").
        if (
            not patch_id
            or patch_id in (".", "..")
            or any(sep in patch_id for sep in ("/", "\\", ":"))
        ):
            raise PatchNotFoundError(f"Invalid patch id: {patch_id!r}")
        return self.patches_path / patch_id

    def _allocate_patch_dir(self, backup_key: str, now: datetime) -> tuple[str, Path]:
        """Create a new, empty patch directory.

        Two patches of one dataset can be created within the same second
        (e.g. a baseline replaced twice in quick succession); a numeric
        suffix keeps the newer patch from overwriting the older one.

        Returns:
            Tuple of (patch_id, patch directory)
        """
        base_id = f"{backup_key}__{now.strftime('%Y-%m-%dT%H-%M-%S')}"
        self.patches_path.mkdir(parents=True, exist_ok=True)

        patch_id = base_id
        counter = 1
        while True:
            patch_dir = self._get_patch_dir(patch_id)
            try:
                # Exclusive create: never reuse an existing patch directory
                patch_dir.mkdir()
                return patch_id, patch_dir
            except FileExistsError:
                counter += 1
                patch_id = f"{base_id}_{counter}"

    def create(
        self,
        dataset: str,
        backup_key: str,
        old_data: pd.DataFrame,
        comparison_result: ComparisonResult,
        old_hash: str,
        new_hash: str,
        reason: str | None = None,
    ) -> PatchMetadata:
        """Create a permanent patch preserving the old baseline.

        Args:
            dataset: Original dataset name
            backup_key: Sanitized backup key
            old_data: The baseline DataFrame being replaced
            comparison_result: Comparison between old and new data
            old_hash: Content hash of old data
            new_hash: Content hash of new data
            reason: Optional reason given when accepting

        Returns:
            Metadata for the created patch
        """
        now = datetime.now()
        patch_id, patch_dir = self._allocate_patch_dir(backup_key, now)

        metadata = PatchMetadata(
            patch_id=patch_id,
            dataset=dataset,
            backup_key=backup_key,
            created_at=now,
            reason=reason,
            old_hash=old_hash,
            new_hash=new_hash,
            old_shape=(len(old_data), len(old_data.columns)),
            new_shape=comparison_result.new_shape,
            diff_summary=_build_diff_summary(comparison_result),
        )

        # Write parquet first, json last: patch.json presence marks a
        # complete patch, so readers skip interrupted writes.
        try:
            old_data.to_parquet(patch_dir / OLD_DATA_NAME, compression=self.compression)
            json_path = patch_dir / PATCH_JSON_NAME
            json_path.write_text(
                json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            shutil.rmtree(patch_dir, ignore_errors=True)
            raise

        logger.info(f"Created permanent patch: {patch_id}")
        return metadata

    def list_patches(self, dataset: str | None = None) -> list[PatchMetadata]:
        """List all patches, optionally filtered by dataset.

        Args:
            dataset: Optional dataset name filter

        Returns:
            Patches sorted by creation time (newest first)
        """
        if not self.patches_path.exists():
            return []

        patches = []
        for patch_dir in self.patches_path.iterdir():
            if not patch_dir.is_dir():
                continue
            json_path = patch_dir / PATCH_JSON_NAME
            if not json_path.exists():
                logger.warning(f"Skipping incomplete patch dir: {patch_dir.name}")
                continue
            try:
                metadata = PatchMetadata.from_dict(
                    json.loads(json_path.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Skipping unreadable patch {patch_dir.name}: {e}")
                continue

            if dataset is not None and metadata.dataset != dataset:
                continue
            patches.append(metadata)

        patches.sort(key=lambda m: m.created_at, reverse=True)
        return patches

    def load_metadata(self, patch_id: str) -> PatchMetadata:
        """Load metadata for a specific patch.

        Args:
            patch_id: The patch identifier

        Returns:
            Patch metadata

        Raises:
            PatchNotFoundError: If patch does not exist
        """
        json_path = self._get_patch_dir(patch_id) / PATCH_JSON_NAME
        if not json_path.exists():
            raise PatchNotFoundError(f"Patch not found: {patch_id}")

        return PatchMetadata.from_dict(
            json.loads(json_path.read_text(encoding="utf-8"))
        )

    def load_old_data(self, patch_id: str) -> pd.DataFrame:
        """Load the preserved old baseline DataFrame for a patch.

        Args:
            patch_id: The patch identifier

        Returns:
            The old baseline DataFrame

        Raises:
            PatchNotFoundError: If patch does not exist
        """
        patch_dir = self._get_patch_dir(patch_id)
        data_path = patch_dir / OLD_DATA_NAME
        if not (patch_dir / PATCH_JSON_NAME).exists() or not data_path.exists():
            raise PatchNotFoundError(f"Patch not found: {patch_id}")

        return pq.read_table(data_path).to_pandas()

    def delete(self, patch_id: str) -> bool:
        """Delete a patch permanently.

        Args:
            patch_id: The patch identifier

        Returns:
            True if deleted, False if patch did not exist
        """
        try:
            patch_dir = self._get_patch_dir(patch_id)
        except PatchNotFoundError:
            return False
        if not patch_dir.exists():
            return False

        for child in patch_dir.iterdir():
            child.unlink()
        patch_dir.rmdir()

        logger.info(f"Deleted patch: {patch_id}")
        return True
