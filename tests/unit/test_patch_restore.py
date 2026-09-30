"""Tests for restoring permanent patches (restore_patch)."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from finlab_sentinel.comparison.differ import DataFrameComparer
from finlab_sentinel.comparison.hasher import ContentHasher
from finlab_sentinel.config.schema import (
    AnomalyBehavior,
    AnomalyConfig,
    SentinelConfig,
    StorageConfig,
)
from finlab_sentinel.core.hooks import clear_preprocess_hooks, register_preprocess_hook
from finlab_sentinel.core.interceptor import DataInterceptor, accept_current_data
from finlab_sentinel.exceptions import (
    DataAnomalyError,
    PatchNotFoundError,
    PatchRestoreError,
)
from finlab_sentinel.storage.backend import BackupMetadata
from finlab_sentinel.storage.index import BackupIndex
from finlab_sentinel.storage.parquet import ParquetStorage
from finlab_sentinel.storage.patches import PatchMetadata, PatchStore, restore_patch

DATASET = "price:收盤價"
KEY = "price__收盤價"
COMPARE = "finlab_sentinel.comparison.differ.DataFrameComparer.compare"


def _price_frame() -> pd.DataFrame:
    """finlab-like price table with a missing value."""
    dates = pd.date_range("2026-09-01", periods=8, name="date")
    data = np.random.default_rng(7).random((8, 4)) * 100 + 500
    df = pd.DataFrame(data, index=dates, columns=["2330", "2317", "1101", "8349"])
    df.iloc[2, 3] = np.nan
    return df


def _mixed_frame() -> pd.DataFrame:
    """Frame covering nullable, object, bool and datetime columns."""
    return pd.DataFrame(
        {
            "close": [580.0, np.nan, 45.2, 101.5],
            "volume": pd.array([1000, None, 300, 42], dtype="Int64"),
            "name": pd.Series(["台積電", None, "台泥", "KY"], dtype=object),
            "halted": [False, True, False, False],
            "listed": pd.to_datetime(
                ["1994-09-05", "1991-01-01", "1962-02-09", "2001-01-01"]
            ),
        },
        index=pd.Index(["2330", "2317", "1101", "8349"], name="stock_id"),
    )


def _revised(df: pd.DataFrame) -> pd.DataFrame:
    """A data revision the append-only policy rejects (last row deleted)."""
    return df.iloc[:-1].copy()


@pytest.fixture
def config(tmp_path: Path) -> SentinelConfig:
    """Config with temporary storage that raises on anomalies."""
    return SentinelConfig(
        storage=StorageConfig(path=tmp_path / "sentinel"),
        anomaly=AnomalyConfig(behavior=AnomalyBehavior.RAISE),
    )


@pytest.fixture
def finlab_frames():
    """Serve DataFrames through a stub ``finlab`` module's data.get."""
    from finlab_sentinel.core import registry

    frames: dict[str, pd.DataFrame] = {}
    data = ModuleType("finlab.data")
    data.get = lambda dataset, *args, **kwargs: frames[dataset].copy()
    finlab = ModuleType("finlab")
    finlab.data = data

    saved = sys.modules.get("finlab")
    sys.modules["finlab"] = finlab
    registry._original_functions.pop("data.get", None)
    yield frames
    if saved is None:
        sys.modules.pop("finlab", None)
    else:
        sys.modules["finlab"] = saved


@pytest.fixture(autouse=True)
def _clear_hooks():
    clear_preprocess_hooks()
    yield
    clear_preprocess_hooks()


@pytest.fixture
def same_second_clock(monkeypatch):
    """Run every storage and patch write within the same wall-clock second."""
    base = datetime(2026, 9, 30, 3, 30, 0)
    ticks = iter(range(1, 1000))

    class SameSecondDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return (base + timedelta(milliseconds=next(ticks))).replace(tzinfo=tz)

    for module in ("parquet", "patches"):
        monkeypatch.setattr(
            f"finlab_sentinel.storage.{module}.datetime", SameSecondDatetime
        )


def _storage(config: SentinelConfig) -> ParquetStorage:
    return ParquetStorage(
        base_path=config.get_storage_path(),
        compression=config.storage.compression,
    )


def _patch_store(config: SentinelConfig) -> PatchStore:
    return PatchStore(base_path=config.get_storage_path())


def _production_get(config: SentinelConfig, frames: dict) -> pd.DataFrame:
    """data.get as production runs it: through the interceptor."""
    interceptor = DataInterceptor(lambda ds, *a, **k: frames[ds].copy(), config)
    return interceptor(DATASET)


def _baseline_then_accept(
    config: SentinelConfig, frames: dict, original: pd.DataFrame
) -> PatchMetadata:
    """Baseline ``original``, hit an anomaly on a revision, accept it."""
    frames[DATASET] = original
    _production_get(config, frames)

    frames[DATASET] = _revised(original)
    with pytest.raises(DataAnomalyError):
        _production_get(config, frames)

    assert accept_current_data(DATASET, config, reason="finlab revision") is True
    patches = _patch_store(config).list_patches(DATASET)
    assert len(patches) == 1
    return patches[0]


def _snapshot(root: Path) -> dict[str, str | None]:
    """Map every path under root to its content digest (None for dirs)."""
    return {
        str(p.relative_to(root)): (
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        )
        for p in sorted(root.rglob("*"))
    }


def _raise(exc: Exception):
    def _fail(*args, **kwargs):
        raise exc

    return _fail


class TestRestoreRoundTrip:
    """restore_patch undoes an accept exactly."""

    @pytest.mark.parametrize(
        "make_frame", [_price_frame, _mixed_frame], ids=["price", "mixed"]
    )
    def test_restore_undoes_accept_exactly(
        self, config, finlab_frames, same_second_clock, make_frame
    ):
        """Stored hash, metadata and data are identical to before the accept."""
        finlab_frames[DATASET] = make_frame()
        _production_get(config, finlab_frames)
        storage = _storage(config)
        before = storage.get_latest_metadata(KEY)
        before_df = storage.load_backup(before)
        before_table = pq.read_table(before.file_path)

        patch_meta = _baseline_then_accept(config, finlab_frames, make_frame())
        assert storage.get_latest_metadata(KEY).content_hash != before.content_hash

        result = restore_patch(patch_meta.patch_id, config)

        after = storage.get_latest_metadata(KEY)
        assert after == result.baseline
        # What the comparison reads: content hash and backup identity
        assert after.content_hash == before.content_hash
        assert (after.dataset, after.backup_key) == (before.dataset, before.backup_key)
        assert (after.row_count, after.column_count) == (
            before.row_count,
            before.column_count,
        )
        # Data: same frame, same raw hash, same parquet columns and pandas schema
        after_df = storage.load_backup(after)
        pd.testing.assert_frame_equal(after_df, before_df, check_exact=True)
        hasher = ContentHasher()
        assert hasher.hash_dataframe(after_df) == hasher.hash_dataframe(before_df)
        after_table = pq.read_table(after.file_path)
        assert after_table.equals(before_table, check_metadata=False)
        after_meta = after_table.schema.metadata
        assert after_meta[b"pandas"] == before_table.schema.metadata[b"pandas"]
        assert after_meta[b"content_hash"] == before.content_hash.encode()
        assert after_meta[b"restored_from_patch"] == patch_meta.patch_id.encode()

    def test_next_run_behaves_as_before_accept(self, config, finlab_frames):
        """The originally baselined data passes; the accepted revision raises."""
        original = _price_frame()
        patch_meta = _baseline_then_accept(config, finlab_frames, original)

        restore_patch(patch_meta.patch_id, config)
        storage = _storage(config)
        backups_after_restore = storage.list_backups(KEY)

        # Original data matches via the hash fast path: no full comparison,
        # nothing saved
        finlab_frames[DATASET] = original
        with patch(COMPARE, side_effect=AssertionError("full comparison ran")):
            returned = _production_get(config, finlab_frames)
        pd.testing.assert_frame_equal(returned, original)
        assert storage.list_backups(KEY) == backups_after_restore

        # The revision that needed the accept is an anomaly again
        finlab_frames[DATASET] = _revised(original)
        with pytest.raises(DataAnomalyError):
            _production_get(config, finlab_frames)

    def test_restores_preprocessed_hash_without_hooks(self, config, finlab_frames):
        """A CLI restore (no hooks) restores the hash production computed."""

        def hook(df):
            return df.round(1)

        original = _price_frame()
        register_preprocess_hook(DATASET, hook)
        finlab_frames[DATASET] = original
        _production_get(config, finlab_frames)
        hooked_hash = _storage(config).get_latest_metadata(KEY).content_hash
        assert hooked_hash != ContentHasher().hash_dataframe(original)

        finlab_frames[DATASET] = _revised(original)
        with pytest.raises(DataAnomalyError):
            _production_get(config, finlab_frames)

        # Accept and restore from a process without the production hooks
        clear_preprocess_hooks()
        assert accept_current_data(DATASET, config) is True
        patch_meta = _patch_store(config).list_patches(DATASET)[0]
        result = restore_patch(patch_meta.patch_id, config)
        assert result.baseline.content_hash == hooked_hash

        # Production, hooks registered, sees the original data as unchanged
        register_preprocess_hook(DATASET, hook)
        finlab_frames[DATASET] = original
        with patch(COMPARE, side_effect=AssertionError("full comparison ran")):
            _production_get(config, finlab_frames)

    def test_restore_can_itself_be_restored(
        self, config, finlab_frames, same_second_clock
    ):
        """Each restore saves the replaced baseline as a restorable patch."""
        original = _price_frame()
        patch_meta = _baseline_then_accept(config, finlab_frames, original)
        storage = _storage(config)
        accepted = storage.get_latest_metadata(KEY)
        accepted_df = storage.load_backup(accepted)

        first = restore_patch(patch_meta.patch_id, config)
        new_patch = first.new_patch
        assert new_patch is not None
        assert new_patch.restored_from == patch_meta.patch_id
        assert new_patch.reason == f"restore of {patch_meta.patch_id}"
        assert new_patch.old_hash == accepted.content_hash
        assert new_patch.new_hash == patch_meta.old_hash
        assert new_patch.diff_summary["added_rows"] == 1

        # Undo the restore: back to the accepted baseline
        second = restore_patch(first.new_patch_id, config)
        latest = storage.get_latest_metadata(KEY)
        assert latest.content_hash == accepted.content_hash
        pd.testing.assert_frame_equal(
            storage.load_backup(latest), accepted_df, check_exact=True
        )

        # And forward again
        third = restore_patch(second.new_patch_id, config)
        assert storage.get_latest_metadata(KEY).content_hash == patch_meta.old_hash
        assert third.previous.content_hash == accepted.content_hash

        # Everything happened within one second: all patches and backup
        # files are distinct and still hold their own data
        patch_ids = [p.patch_id for p in _patch_store(config).list_patches(DATASET)]
        assert len(set(patch_ids)) == 4
        backups = storage.list_backups(KEY)
        assert len({b.file_path for b in backups}) == len(backups)
        for backup in backups:
            stored_hash = pq.read_table(backup.file_path).schema.metadata[
                b"content_hash"
            ]
            assert stored_hash == backup.content_hash.encode()


class TestRestoreBehavior:
    """Edge cases and guarantees of restore_patch."""

    def test_missing_patch_raises_and_changes_nothing(self, config, finlab_frames):
        _baseline_then_accept(config, finlab_frames, _price_frame())
        root = config.get_storage_path()
        before = _snapshot(root)

        with pytest.raises(PatchNotFoundError):
            restore_patch(f"{KEY}__2000-01-01T00-00-00", config)
        with pytest.raises(PatchNotFoundError):
            restore_patch("../data", config)

        assert _snapshot(root) == before

    def test_restore_without_baseline(self, config, sample_df, sample_df_modified):
        """A dataset without any backup simply gets the patch data."""
        store = _patch_store(config)
        patch_meta = store.create(
            dataset=DATASET,
            backup_key=KEY,
            old_data=sample_df,
            comparison_result=DataFrameComparer().compare(
                sample_df, sample_df_modified
            ),
            old_hash="old-hash",
            new_hash="new-hash",
        )

        result = restore_patch(patch_meta.patch_id, config)

        assert result.changed
        assert result.previous is None
        assert result.new_patch is None
        latest = _storage(config).get_latest_metadata(KEY)
        assert latest == result.baseline
        assert latest.content_hash == "old-hash"
        pd.testing.assert_frame_equal(
            _storage(config).load_backup(latest), sample_df, check_freq=False
        )
        assert store.list_patches() == [patch_meta]

    def test_restore_when_baseline_file_is_missing(self, config, finlab_frames):
        """A baseline whose file is gone is replaced without a patch."""
        patch_meta = _baseline_then_accept(config, finlab_frames, _price_frame())
        storage = _storage(config)
        storage.get_latest_metadata(KEY).file_path.unlink()

        result = restore_patch(patch_meta.patch_id, config)

        assert result.previous is None
        assert result.new_patch is None
        latest = storage.get_latest_metadata(KEY)
        assert latest == result.baseline
        assert latest.content_hash == patch_meta.old_hash
        assert storage.load_latest(KEY) is not None

    def test_dry_run_writes_nothing(self, config, finlab_frames):
        patch_meta = _baseline_then_accept(config, finlab_frames, _price_frame())
        accepted = _storage(config).get_latest_metadata(KEY)
        root = config.get_storage_path()
        before = _snapshot(root)

        result = restore_patch(patch_meta.patch_id, config, dry_run=True)

        assert _snapshot(root) == before
        assert result.dry_run
        assert not result.changed
        assert not result.already_current
        assert result.new_patch is None
        assert result.previous == accepted
        assert result.patch == patch_meta

    def test_dry_run_without_backups_creates_nothing(self, config, sample_df):
        store = _patch_store(config)
        patch_meta = store.create(
            dataset=DATASET,
            backup_key=KEY,
            old_data=sample_df,
            comparison_result=DataFrameComparer().compare(sample_df, sample_df),
            old_hash="old-hash",
            new_hash="new-hash",
        )
        root = config.get_storage_path()
        before = _snapshot(root)

        result = restore_patch(patch_meta.patch_id, config, dry_run=True)

        assert _snapshot(root) == before
        assert not (root / "data").exists()
        assert result.previous is None

    def test_restoring_twice_is_a_no_op(self, config, finlab_frames):
        patch_meta = _baseline_then_accept(config, finlab_frames, _price_frame())
        restore_patch(patch_meta.patch_id, config)
        root = config.get_storage_path()
        before = _snapshot(root)

        result = restore_patch(patch_meta.patch_id, config)

        assert result.already_current
        assert not result.changed
        assert result.new_patch is None
        assert _snapshot(root) == before

    def test_source_patch_is_kept_and_reason_recorded(self, config, finlab_frames):
        original = _price_frame()
        patch_meta = _baseline_then_accept(config, finlab_frames, original)

        result = restore_patch(
            patch_meta.patch_id, config, reason="xstock-restore 20260930-0330"
        )

        store = _patch_store(config)
        assert store.load_metadata(patch_meta.patch_id) == patch_meta
        preserved = store.load_old_data(patch_meta.patch_id)
        pd.testing.assert_frame_equal(preserved, original, check_freq=False)
        new_patch = store.load_metadata(result.new_patch_id)
        assert new_patch.reason == "xstock-restore 20260930-0330"
        assert new_patch.restored_from == patch_meta.patch_id

    def test_restored_baseline_survives_retention(
        self, config, finlab_frames, same_second_clock
    ):
        """The restored baseline is kept like any freshly saved baseline."""
        original = _price_frame()
        patch_meta = _baseline_then_accept(config, finlab_frames, original)
        restore_patch(patch_meta.patch_id, config)
        storage = _storage(config)
        restored = storage.get_latest_metadata(KEY)

        # Expire every older backup of the dataset
        older = [b for b in storage.list_backups(KEY) if b != restored]
        assert older
        long_ago = datetime.now() - timedelta(days=30)
        with storage.index._connect() as conn:
            for i, backup in enumerate(older):
                conn.execute(
                    "UPDATE backups SET created_at = ? WHERE file_path = ?",
                    (
                        (long_ago - timedelta(minutes=i)).isoformat(),
                        str(backup.file_path),
                    ),
                )
        storage.cleanup_expired(retention_days=7, min_keep_per_key=1)

        assert storage.list_backups(KEY) == [restored]
        loaded, metadata = storage.load_latest(KEY)
        assert metadata == restored
        pd.testing.assert_frame_equal(loaded, original, check_freq=False)

    @pytest.mark.parametrize("fail_at", ["patch", "write", "fsync", "index"])
    def test_failure_leaves_baseline_unchanged(
        self, config, finlab_frames, monkeypatch, fail_at
    ):
        patch_meta = _baseline_then_accept(config, finlab_frames, _price_frame())
        accepted = _storage(config).get_latest_metadata(KEY)
        root = config.get_storage_path()
        before = _snapshot(root)

        target = {
            "patch": (PatchStore, "create", OSError("disk full")),
            "write": (ParquetStorage, "_write_backup", OSError("disk full")),
            "fsync": (
                sys.modules["finlab_sentinel.storage.parquet"],
                "_fsync_file",
                OSError("I/O error"),
            ),
            "index": (
                BackupIndex,
                "add_if_latest",
                sqlite3.OperationalError("database is locked"),
            ),
        }[fail_at]
        monkeypatch.setattr(target[0], target[1], _raise(target[2]))

        with pytest.raises(PatchRestoreError):
            restore_patch(patch_meta.patch_id, config)

        assert _storage(config).get_latest_metadata(KEY) == accepted
        # No leftover patch, backup file or index entry
        assert _snapshot(root) == before

    def test_concurrent_baseline_change_aborts(
        self, config, finlab_frames, monkeypatch
    ):
        """A baseline saved while restoring is never silently superseded."""
        original = _price_frame()
        patch_meta = _baseline_then_accept(config, finlab_frames, original)
        concurrent: dict[str, BackupMetadata] = {}
        real_add_if_latest = BackupIndex.add_if_latest

        def racing_add_if_latest(self, metadata, expected_latest, reason=None):
            # Another process saves a new baseline just before the swap
            concurrent["backup"] = _storage(config).save(
                KEY, DATASET, original, "concurrent-hash"
            )
            return real_add_if_latest(self, metadata, expected_latest, reason=reason)

        monkeypatch.setattr(BackupIndex, "add_if_latest", racing_add_if_latest)

        with pytest.raises(PatchRestoreError, match="changed"):
            restore_patch(patch_meta.patch_id, config)

        storage = _storage(config)
        assert storage.get_latest_metadata(KEY) == concurrent["backup"]
        patches = _patch_store(config).list_patches(DATASET)
        assert [p.patch_id for p in patches] == [patch_meta.patch_id]
        backup_files = set(storage._get_backup_dir(KEY).iterdir())
        assert backup_files == {b.file_path for b in storage.list_backups(KEY)}

    def test_future_dated_baseline_is_still_superseded(self, config, finlab_frames):
        """Clock skew cannot keep the restored baseline from becoming latest."""
        patch_meta = _baseline_then_accept(config, finlab_frames, _price_frame())
        storage = _storage(config)
        accepted = storage.get_latest_metadata(KEY)
        future = datetime.now() + timedelta(days=1)
        with storage.index._connect() as conn:
            conn.execute(
                "UPDATE backups SET created_at = ? WHERE file_path = ?",
                (future.isoformat(), str(accepted.file_path)),
            )

        result = restore_patch(patch_meta.patch_id, config)

        latest = storage.get_latest_metadata(KEY)
        assert latest == result.baseline
        assert latest.content_hash == patch_meta.old_hash
        assert latest.created_at > future

    def test_inconsistent_patch_is_refused(self, config, finlab_frames):
        patch_meta = _baseline_then_accept(config, finlab_frames, _price_frame())
        json_path = config.get_storage_path() / "patches" / patch_meta.patch_id
        json_path = json_path / "patch.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        data["old_shape"] = [999, 4]
        json_path.write_text(json.dumps(data), encoding="utf-8")
        root = config.get_storage_path()
        before = _snapshot(root)

        with pytest.raises(PatchRestoreError, match="shape"):
            restore_patch(patch_meta.patch_id, config)

        assert _snapshot(root) == before

    def test_unreadable_baseline_is_not_replaced(self, config, finlab_frames):
        patch_meta = _baseline_then_accept(config, finlab_frames, _price_frame())
        _storage(config).get_latest_metadata(KEY).file_path.write_bytes(b"corrupt")
        root = config.get_storage_path()
        before = _snapshot(root)

        with pytest.raises(PatchRestoreError, match="current baseline"):
            restore_patch(patch_meta.patch_id, config)

        assert _snapshot(root) == before

    def test_result_to_dict_is_json_serializable(self, config, finlab_frames):
        patch_meta = _baseline_then_accept(config, finlab_frames, _price_frame())
        accepted = _storage(config).get_latest_metadata(KEY)

        result = restore_patch(patch_meta.patch_id, config)
        data = json.loads(json.dumps(result.to_dict()))

        assert data["patch_id"] == patch_meta.patch_id
        assert data["dataset"] == DATASET
        assert data["restored_hash"] == patch_meta.old_hash
        assert data["changed"] is True
        assert data["dry_run"] is False
        assert data["new_patch_id"] == result.new_patch_id
        assert data["previous"]["content_hash"] == accepted.content_hash
        assert data["baseline"]["content_hash"] == patch_meta.old_hash


class TestPublicAPI:
    """finlab_sentinel.restore_patch at package level."""

    def test_restore_patch_with_default_config(
        self, config, finlab_frames, monkeypatch
    ):
        import finlab_sentinel as fs

        patch_meta = _baseline_then_accept(config, finlab_frames, _price_frame())
        monkeypatch.setattr(
            "finlab_sentinel.config.loader.load_config", lambda *a, **k: config
        )

        result = fs.restore_patch(patch_meta.patch_id)

        assert result.changed
        assert result.baseline.content_hash == patch_meta.old_hash
        assert fs.PatchRestoreError is PatchRestoreError

    def test_restore_patch_not_found(self, config):
        import finlab_sentinel as fs

        with pytest.raises(fs.PatchNotFoundError):
            fs.restore_patch("nonexistent__2026-01-01T00-00-00", config)


class TestConcurrentWrites:
    """Restore and data.get never supersede a baseline they did not read."""

    def test_data_get_does_not_overwrite_concurrent_restore(
        self, config, finlab_frames, caplog
    ):
        """A restore committed while data.get compares stays the baseline."""
        original = _price_frame()
        patch_meta = _baseline_then_accept(config, finlab_frames, original)
        accepted = _revised(original)
        # Today's data: the accepted revision plus one new day (in policy)
        next_day = accepted.index[-1] + pd.Timedelta(days=1)
        finlab_frames[DATASET] = pd.concat(
            [accepted, accepted.iloc[[-1]].set_axis([next_day])]
        )
        interceptor = DataInterceptor(
            lambda ds, *a, **k: finlab_frames[ds].copy(), config
        )
        real_compare = interceptor.comparer.compare

        def compare_then_restore(old, new):
            # The user restores the patch while data.get is comparing
            restore_patch(patch_meta.patch_id, config)
            return real_compare(old, new)

        interceptor.comparer.compare = compare_then_restore
        with caplog.at_level(logging.WARNING, logger="finlab_sentinel"):
            interceptor(DATASET)

        latest = _storage(config).get_latest_metadata(KEY)
        assert latest.content_hash == patch_meta.old_hash
        assert "changed while data.get compared" in caplog.text

    def test_restore_holds_write_lock_between_check_and_insert(
        self, config, finlab_frames, monkeypatch
    ):
        """No other writer can slip in between the latest check and insert."""
        patch_meta = _baseline_then_accept(config, finlab_frames, _price_frame())
        index_path = config.get_storage_path() / "data" / "index.sqlite"
        real_row_to_metadata = BackupIndex._row_to_metadata
        real_add_if_latest = BackupIndex.add_if_latest
        state: dict[str, object] = {"armed": False}

        def racing_row_to_metadata(row):
            if state["armed"]:
                state["armed"] = False
                conn = sqlite3.connect(index_path, timeout=0.1)
                try:
                    conn.execute(
                        "INSERT INTO backups (backup_key, dataset, file_path,"
                        " content_hash, created_at, row_count, column_count,"
                        " file_size_bytes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            KEY,
                            DATASET,
                            "/elsewhere/concurrent.parquet",
                            "concurrent",
                            datetime.now().isoformat(),
                            1,
                            1,
                            1,
                        ),
                    )
                    conn.commit()
                    state["competing"] = "committed"
                except sqlite3.OperationalError as e:
                    state["competing"] = str(e)
                finally:
                    conn.close()
            return real_row_to_metadata(row)

        def armed_add_if_latest(self, *args, **kwargs):
            state["armed"] = True
            return real_add_if_latest(self, *args, **kwargs)

        monkeypatch.setattr(
            BackupIndex, "_row_to_metadata", staticmethod(racing_row_to_metadata)
        )
        monkeypatch.setattr(BackupIndex, "add_if_latest", armed_add_if_latest)

        result = restore_patch(patch_meta.patch_id, config)

        assert state["competing"] == "database is locked"
        assert _storage(config).get_latest_metadata(KEY) == result.baseline
