"""Tests for permanent patch storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from finlab_sentinel.comparison.differ import DataFrameComparer
from finlab_sentinel.exceptions import PatchNotFoundError
from finlab_sentinel.storage.patches import PatchMetadata, PatchStore


@pytest.fixture
def patch_store(tmp_storage: Path) -> PatchStore:
    """Create PatchStore instance for testing."""
    return PatchStore(base_path=tmp_storage)


@pytest.fixture
def comparison_result(sample_df, sample_df_modified):
    """Create a real comparison result between sample DataFrames."""
    comparer = DataFrameComparer()
    return comparer.compare(sample_df, sample_df_modified)


def _create_patch(
    patch_store: PatchStore,
    sample_df: pd.DataFrame,
    comparison_result,
    reason: str | None = "test reason",
) -> PatchMetadata:
    return patch_store.create(
        dataset="price:收盤價",
        backup_key="price__收盤價",
        old_data=sample_df,
        comparison_result=comparison_result,
        old_hash="oldhash123",
        new_hash="newhash456",
        reason=reason,
    )


class TestPatchStoreCreate:
    """Tests for PatchStore.create."""

    def test_create_returns_metadata(
        self, patch_store, sample_df, comparison_result
    ) -> None:
        metadata = _create_patch(patch_store, sample_df, comparison_result)

        assert metadata.dataset == "price:收盤價"
        assert metadata.backup_key == "price__收盤價"
        assert metadata.reason == "test reason"
        assert metadata.old_hash == "oldhash123"
        assert metadata.new_hash == "newhash456"
        assert metadata.patch_id.startswith("price__收盤價__")
        assert metadata.old_shape == (10, 4)
        assert metadata.diff_summary["modified_cells"] == 2

    def test_create_writes_files(
        self, patch_store, sample_df, comparison_result, tmp_storage
    ) -> None:
        metadata = _create_patch(patch_store, sample_df, comparison_result)

        patch_dir = tmp_storage / "patches" / metadata.patch_id
        assert (patch_dir / "old_data.parquet").exists()
        assert (patch_dir / "patch.json").exists()

    def test_create_without_reason(
        self, patch_store, sample_df, comparison_result
    ) -> None:
        metadata = _create_patch(patch_store, sample_df, comparison_result, reason=None)
        assert metadata.reason is None


class TestPatchStoreList:
    """Tests for PatchStore.list_patches."""

    def test_list_empty(self, patch_store) -> None:
        assert patch_store.list_patches() == []

    def test_list_returns_created_patches(
        self, patch_store, sample_df, comparison_result
    ) -> None:
        created = _create_patch(patch_store, sample_df, comparison_result)

        patches = patch_store.list_patches()
        assert len(patches) == 1
        assert patches[0].patch_id == created.patch_id
        assert patches[0].dataset == "price:收盤價"

    def test_list_filters_by_dataset(
        self, patch_store, sample_df, comparison_result
    ) -> None:
        _create_patch(patch_store, sample_df, comparison_result)

        assert len(patch_store.list_patches(dataset="price:收盤價")) == 1
        assert patch_store.list_patches(dataset="price:開盤價") == []

    def test_list_skips_incomplete_patch_dir(
        self, patch_store, sample_df, comparison_result, tmp_storage
    ) -> None:
        _create_patch(patch_store, sample_df, comparison_result)

        # Simulate interrupted write: directory without patch.json
        incomplete = tmp_storage / "patches" / "broken__2026-01-01T00-00-00"
        incomplete.mkdir(parents=True)
        (incomplete / "old_data.parquet").write_bytes(b"junk")

        patches = patch_store.list_patches()
        assert len(patches) == 1


class TestPatchStoreLoad:
    """Tests for loading patch metadata and data."""

    def test_load_metadata(self, patch_store, sample_df, comparison_result) -> None:
        created = _create_patch(patch_store, sample_df, comparison_result)

        loaded = patch_store.load_metadata(created.patch_id)
        assert loaded.patch_id == created.patch_id
        assert loaded.dataset == created.dataset
        assert loaded.diff_summary == created.diff_summary

    def test_load_old_data_roundtrip(
        self, patch_store, sample_df, comparison_result
    ) -> None:
        created = _create_patch(patch_store, sample_df, comparison_result)

        df = patch_store.load_old_data(created.patch_id)
        # Index freq attribute is not preserved by parquet
        pd.testing.assert_frame_equal(df, sample_df, check_freq=False)

    def test_load_metadata_not_found(self, patch_store) -> None:
        with pytest.raises(PatchNotFoundError):
            patch_store.load_metadata("nonexistent__2026-01-01T00-00-00")

    def test_load_old_data_not_found(self, patch_store) -> None:
        with pytest.raises(PatchNotFoundError):
            patch_store.load_old_data("nonexistent__2026-01-01T00-00-00")


class TestPublicAPI:
    """Tests for package-level patch API."""

    def test_list_patches_and_load_patch_data(
        self, patch_store, sample_df, comparison_result, tmp_storage, monkeypatch
    ) -> None:
        import finlab_sentinel as fs
        from finlab_sentinel.config.schema import SentinelConfig, StorageConfig

        created = _create_patch(patch_store, sample_df, comparison_result)

        config = SentinelConfig(storage=StorageConfig(path=tmp_storage))
        monkeypatch.setattr(
            "finlab_sentinel.config.loader.load_config", lambda *a, **k: config
        )

        patches = fs.list_patches()
        assert len(patches) == 1
        assert patches[0].patch_id == created.patch_id

        assert fs.list_patches(dataset="price:開盤價") == []

        df = fs.load_patch_data(created.patch_id)
        pd.testing.assert_frame_equal(df, sample_df, check_freq=False)

    def test_load_patch_data_not_found(self, tmp_storage, monkeypatch) -> None:
        import finlab_sentinel as fs
        from finlab_sentinel.config.schema import SentinelConfig, StorageConfig

        config = SentinelConfig(storage=StorageConfig(path=tmp_storage))
        monkeypatch.setattr(
            "finlab_sentinel.config.loader.load_config", lambda *a, **k: config
        )

        with pytest.raises(PatchNotFoundError):
            fs.load_patch_data("nonexistent__2026-01-01T00-00-00")


class TestPatchStoreDelete:
    """Tests for PatchStore.delete."""

    def test_delete_existing(
        self, patch_store, sample_df, comparison_result, tmp_storage
    ) -> None:
        created = _create_patch(patch_store, sample_df, comparison_result)

        assert patch_store.delete(created.patch_id) is True
        assert patch_store.list_patches() == []
        assert not (tmp_storage / "patches" / created.patch_id).exists()

    def test_delete_nonexistent(self, patch_store) -> None:
        assert patch_store.delete("nonexistent__2026-01-01T00-00-00") is False
