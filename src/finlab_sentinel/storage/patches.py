"""Permanent patch storage for accepted data changes.

When a user accepts anomalous data as the new baseline, a permanent patch
is created preserving the old baseline snapshot and a diff summary. Patches
live in their own directory and are never touched by retention cleanup.

A patch can be restored, making its preserved data the dataset's baseline
again. The baseline it replaces is saved as a new patch, so a restore can
itself be restored.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
import pyarrow.parquet as pq

from finlab_sentinel.comparison.differ import ComparisonResult, DataFrameComparer
from finlab_sentinel.comparison.hasher import ContentHasher
from finlab_sentinel.core.hooks import get_registry as get_preprocess_registry
from finlab_sentinel.exceptions import PatchNotFoundError, PatchRestoreError
from finlab_sentinel.storage.backend import UNSET, BackupMetadata, _Unset
from finlab_sentinel.storage.parquet import (
    ParquetStorage,
    _fsync_dir,
    _fsync_file,
    get_index_path,
    is_direct_child,
    is_valid_backup_key,
)

if TYPE_CHECKING:
    from finlab_sentinel.config.schema import SentinelConfig

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
    # Set when the patch preserves a baseline replaced by restoring this patch
    restored_from: str | None = None

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
            "restored_from": self.restored_from,
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
            restored_from=data.get("restored_from"),
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
        patch_dir = self.patches_path / patch_id
        # Patch ids are single path components; reject anything that would
        # resolve outside the patches directory (e.g. "..", "../data").
        if not is_direct_child(patch_dir, self.patches_path):
            raise PatchNotFoundError(f"Invalid patch id: {patch_id!r}")
        return patch_dir

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
        comparison_result: ComparisonResult | None,
        old_hash: str,
        new_hash: str,
        reason: str | None = None,
        restored_from: str | None = None,
        new_shape: tuple[int, int] | None = None,
        diff_error: str | None = None,
    ) -> PatchMetadata:
        """Create a permanent patch preserving the old baseline.

        The patch files are flushed to disk before this returns, so the
        baseline replaced next stays recoverable even after a crash.

        Args:
            dataset: Original dataset name
            backup_key: Sanitized backup key
            old_data: The baseline DataFrame being replaced
            comparison_result: Comparison between old and new data, or None
                if it could not be computed (pass ``new_shape`` then)
            old_hash: Content hash of old data
            new_hash: Content hash of new data
            reason: Optional reason given when accepting
            restored_from: Patch being restored, when the old baseline is
                replaced by a restore rather than an accept
            new_shape: Shape of the new data, used without a comparison result
            diff_error: Why the comparison failed, recorded in the summary

        Returns:
            Metadata for the created patch
        """
        # A summary that cannot be built must not stop the data being kept
        diff_summary: dict | None = None
        if comparison_result is not None:
            new_shape = comparison_result.new_shape
            try:
                diff_summary = _build_diff_summary(comparison_result)
            except Exception as e:
                diff_error = f"{type(e).__name__}: {e}"
        if diff_summary is None:
            diff_summary = {"summary_text": f"diff unavailable: {diff_error}"}
        if new_shape is None:
            raise ValueError("new_shape is required without a comparison result")

        now = datetime.now()
        patch_id, patch_dir = self._allocate_patch_dir(backup_key, now)

        # Write parquet first, json last: patch.json presence marks a
        # complete patch, so readers skip interrupted writes.
        try:
            metadata = PatchMetadata(
                patch_id=patch_id,
                dataset=dataset,
                backup_key=backup_key,
                created_at=now,
                reason=reason,
                old_hash=old_hash,
                new_hash=new_hash,
                old_shape=(len(old_data), len(old_data.columns)),
                new_shape=new_shape,
                diff_summary=diff_summary,
                restored_from=restored_from,
            )
            data_path = patch_dir / OLD_DATA_NAME
            old_data.to_parquet(data_path, compression=self.compression)
            _fsync_file(data_path)
            json_path = patch_dir / PATCH_JSON_NAME
            json_path.write_text(
                json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _fsync_file(json_path)
            _fsync_dir(self.patches_path)
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


@dataclass
class RestoreResult:
    """Outcome of restoring a patch (see restore_patch).

    Attributes:
        patch: The restored (source) patch; it is kept, not consumed
        previous: The dataset's baseline before the restore (the one that
            was, or for a dry run would be, replaced), or None if the dataset
            had no usable baseline
        latest: The latest index entry the restore was planned against;
            equals ``previous`` unless that entry's file is missing. Pass it
            as ``expected_latest`` to restore only if nothing changed since
        baseline: The new baseline written by the restore; None for a dry
            run or when the baseline already matched the patch
        new_patch: Patch preserving ``previous`` so the restore can itself be
            restored; None if there was no previous baseline, for a dry run,
            or when nothing changed
        already_current: The baseline already equals the patch (same content
            hash and data), so nothing was or would be written
        dry_run: Whether this was a dry run (nothing written)
    """

    patch: PatchMetadata
    previous: BackupMetadata | None
    latest: BackupMetadata | None
    baseline: BackupMetadata | None
    new_patch: PatchMetadata | None
    already_current: bool
    dry_run: bool

    @property
    def changed(self) -> bool:
        """Whether the dataset's baseline was replaced."""
        return self.baseline is not None

    @property
    def new_patch_id(self) -> str | None:
        """ID of the patch preserving the replaced baseline, if one was made."""
        return self.new_patch.patch_id if self.new_patch is not None else None

    @property
    def baseline_file_missing(self) -> bool:
        """Whether the latest index entry exists but its file is gone."""
        return self.previous is None and self.latest is not None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "patch_id": self.patch.patch_id,
            "dataset": self.patch.dataset,
            "backup_key": self.patch.backup_key,
            "restored_hash": self.patch.old_hash,
            "dry_run": self.dry_run,
            "already_current": self.already_current,
            "changed": self.changed,
            "previous": self.previous.to_dict() if self.previous else None,
            "latest": self.latest.to_dict() if self.latest else None,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "new_patch_id": self.new_patch_id,
        }


def preserve_baseline(
    patch_store: PatchStore,
    config: SentinelConfig,
    dataset: str,
    backup_key: str,
    old_data: pd.DataFrame,
    new_data: pd.DataFrame,
    old_hash: str,
    new_hash: str,
    reason: str | None = None,
    restored_from: str | None = None,
) -> PatchMetadata:
    """Save a baseline that is about to be replaced as a permanent patch.

    The diff summary compares the way DataInterceptor does, after preprocess
    hooks (which get copies, so ``old_data`` is written untouched). If the
    comparison fails, e.g. on a duplicate index, the patch is still created
    with a placeholder summary: keeping the data matters more than the diff.

    Args:
        patch_store: Store to create the patch in
        config: Configuration (comparison tolerances)
        dataset: Dataset name
        backup_key: Backup key of the dataset
        old_data: The baseline being replaced
        new_data: The data replacing it
        old_hash: Content hash of the baseline being replaced
        new_hash: Content hash of the replacing data
        reason: Reason recorded on the patch
        restored_from: Patch being restored, when a restore replaces it

    Returns:
        Metadata of the created patch
    """
    registry = get_preprocess_registry()
    comparer = DataFrameComparer(
        rtol=config.comparison.rtol,
        atol=config.comparison.atol,
        check_dtype=config.comparison.check_dtype,
        check_na_type=config.comparison.check_na_type,
    )
    result: ComparisonResult | None = None
    diff_error: str | None = None
    try:
        result = comparer.compare(
            registry.apply(dataset, old_data.copy()),
            registry.apply(dataset, new_data.copy()),
        )
    except Exception as e:
        diff_error = f"{type(e).__name__}: {e}"
        logger.warning(f"Could not diff {dataset} for its patch: {diff_error}")

    return patch_store.create(
        dataset=dataset,
        backup_key=backup_key,
        old_data=old_data,
        comparison_result=result,
        old_hash=old_hash,
        new_hash=new_hash,
        reason=reason,
        restored_from=restored_from,
        new_shape=(len(new_data), len(new_data.columns)),
        diff_error=diff_error,
    )


def restore_patch(
    patch_id: str,
    config: SentinelConfig | None = None,
    reason: str | None = None,
    dry_run: bool = False,
    expected_latest: BackupMetadata | None | _Unset = UNSET,
) -> RestoreResult:
    """Make the baseline preserved by a patch the dataset's baseline again.

    The patch's data and its recorded content hash (``old_hash``) become the
    latest baseline of the patch's backup key, exactly as they were before
    the accept that created the patch, so the next data.get comparison
    behaves as it did before that accept. The hash is copied, never
    recomputed: DataInterceptor stores it over preprocessed data, and the
    caller (e.g. the CLI) may not have the same preprocess hooks registered.

    Before the baseline is replaced it is saved as a new patch (with
    ``restored_from`` set to ``patch_id``), so the restore can itself be
    restored. The source patch is kept. The restored baseline is a regular
    backup created at restore time, so retention cleanup treats it like any
    freshly saved baseline.

    Restoring is idempotent: if the baseline already equals the patch,
    nothing is written.

    Do not restore a dataset while a sentinel-enabled process may be using
    it. Baseline writes are compare-and-swap, so neither side overwrites the
    other (this restore aborts if the baseline changes underneath it, and a
    data.get that compared against the old baseline does not save over the
    restored one), but such a process keeps the data it already validated
    against the old baseline.

    Args:
        patch_id: The patch identifier (see list_patches)
        config: Optional configuration (uses default if not provided)
        reason: Reason recorded on the new patch and the restored baseline
            (default: "restore of <patch_id>")
        dry_run: Only report what would change, without writing anything
        expected_latest: If given (typically ``latest`` from a dry run),
            abort unless the dataset's latest backup is still this entry

    Returns:
        RestoreResult describing the previous and restored baselines

    Raises:
        PatchNotFoundError: If the patch does not exist
        PatchRestoreError: If the restore cannot be completed; the dataset's
            baseline is left unchanged
    """
    if config is None:
        from finlab_sentinel.config.loader import load_config

        config = load_config()

    storage_path = config.get_storage_path()
    patch_store = PatchStore(
        base_path=storage_path,
        compression=config.storage.compression,
    )

    try:
        patch = patch_store.load_metadata(patch_id)
        patch_data = patch_store.load_old_data(patch_id)
    except PatchNotFoundError:
        raise
    except Exception as e:
        raise PatchRestoreError(f"Cannot read patch {patch_id}: {e}") from e

    if not is_valid_backup_key(patch.backup_key):
        raise PatchRestoreError(
            f"Patch {patch_id} has an invalid backup key: {patch.backup_key!r}"
        )
    shape = (len(patch_data), len(patch_data.columns))
    if shape != tuple(patch.old_shape):
        raise PatchRestoreError(
            f"Data of patch {patch_id} has shape {shape}, "
            f"but the patch records {tuple(patch.old_shape)}"
        )

    # A dry run must not create an index for a storage that has none
    storage: ParquetStorage | None = None
    latest: BackupMetadata | None = None
    try:
        if not dry_run or get_index_path(storage_path).exists():
            storage = ParquetStorage(
                base_path=storage_path,
                compression=config.storage.compression,
            )
            latest = storage.get_latest_metadata(patch.backup_key)
    except Exception as e:
        raise PatchRestoreError(
            f"Cannot read the backup index in {storage_path}: {e}"
        ) from e

    if expected_latest is not UNSET and latest != expected_latest:
        raise PatchRestoreError(
            f"Baseline of {patch.dataset} changed since it was read; nothing restored"
        )

    previous: BackupMetadata | None = None
    previous_data: pd.DataFrame | None = None
    if storage is not None and latest is not None:
        try:
            previous_data = storage.load_backup(latest)
        except Exception as e:
            raise PatchRestoreError(
                f"Cannot read current baseline of {patch.dataset} "
                f"({latest.file_path}); refusing to replace it: {e}"
            ) from e
        if previous_data is None:
            # DataInterceptor treats this as "no baseline" as well
            logger.warning(
                f"Baseline file of {patch.dataset} is missing "
                f"({latest.file_path}); nothing to preserve before restoring"
            )
        else:
            previous = latest

    # Same hash is not enough: a preprocess hook can hide raw differences
    hasher = ContentHasher()
    already_current = (
        previous is not None
        and previous_data is not None
        and previous.content_hash == patch.old_hash
        and hasher.hash_dataframe(previous_data) == hasher.hash_dataframe(patch_data)
    )

    if dry_run or already_current:
        if already_current:
            logger.info(f"Baseline of {patch.dataset} already matches patch {patch_id}")
        return RestoreResult(
            patch=patch,
            previous=previous,
            latest=latest,
            baseline=None,
            new_patch=None,
            already_current=already_current,
            dry_run=dry_run,
        )

    assert storage is not None  # always opened when not a dry run
    effective_reason = reason or f"restore of {patch_id}"

    # Preserve the current baseline first; never replace what cannot be kept
    new_patch: PatchMetadata | None = None
    if previous is not None and previous_data is not None:
        try:
            new_patch = preserve_baseline(
                patch_store,
                config,
                dataset=patch.dataset,
                backup_key=patch.backup_key,
                old_data=previous_data,
                new_data=patch_data,
                old_hash=previous.content_hash,
                new_hash=patch.old_hash,
                reason=effective_reason,
                restored_from=patch_id,
            )
        except Exception as e:
            raise PatchRestoreError(
                f"Cannot save current baseline of {patch.dataset} as a patch "
                f"(baseline left unchanged): {e}"
            ) from e

    try:
        baseline = storage.restore_baseline(
            backup_key=patch.backup_key,
            dataset=patch.dataset,
            data=patch_data,
            content_hash=patch.old_hash,
            patch_id=patch_id,
            expected_latest=latest,
            reason=effective_reason,
        )
    except Exception as e:
        if new_patch is not None:
            with contextlib.suppress(Exception):
                patch_store.delete(new_patch.patch_id)
        raise PatchRestoreError(
            f"Failed to restore patch {patch_id} (baseline left unchanged): {e}"
        ) from e

    logger.info(
        f"Restored baseline of {patch.dataset} from patch {patch_id}"
        + (
            f"; previous baseline saved as patch {new_patch.patch_id}"
            if new_patch is not None
            else ""
        )
    )

    return RestoreResult(
        patch=patch,
        previous=previous,
        latest=latest,
        baseline=baseline,
        new_patch=new_patch,
        already_current=False,
        dry_run=False,
    )
