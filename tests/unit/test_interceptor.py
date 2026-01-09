"""Tests for data interception and orchestration."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from finlab_sentinel.config.schema import (
    AnomalyBehavior,
    AnomalyConfig,
    ComparisonConfig,
    ComparisonPoliciesConfig,
    PolicyMode,
    SentinelConfig,
    StorageConfig,
)
from finlab_sentinel.core.interceptor import DataInterceptor, accept_current_data
from finlab_sentinel.exceptions import DataAnomalyError


@pytest.fixture
def config_for_interceptor(tmp_path: Path) -> SentinelConfig:
    """Create config for interceptor tests."""
    return SentinelConfig(
        storage=StorageConfig(path=tmp_path),
        comparison=ComparisonConfig(
            policies=ComparisonPoliciesConfig(default_mode=PolicyMode.APPEND_ONLY)
        ),
        anomaly=AnomalyConfig(behavior=AnomalyBehavior.RAISE),
    )


@pytest.fixture
def mock_data_get():
    """Create a mock data.get function."""

    def _get(dataset: str, *args, **kwargs):
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    return MagicMock(side_effect=_get)


class TestDataInterceptor:
    """Tests for DataInterceptor class."""

    def test_first_call_saves_baseline(
        self, config_for_interceptor: SentinelConfig, mock_data_get
    ):
        """Verify first call saves data as baseline."""
        interceptor = DataInterceptor(mock_data_get, config_for_interceptor)

        result = interceptor("test:dataset")

        # Should return the data
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

        # Should have saved baseline
        cached = interceptor.storage.load_latest("test__dataset")
        assert cached is not None

    def test_identical_data_returns_new(
        self, config_for_interceptor: SentinelConfig, mock_data_get
    ):
        """Verify identical data returns new data without issues."""
        interceptor = DataInterceptor(mock_data_get, config_for_interceptor)

        # First call - baseline
        interceptor("test:dataset")

        # Second call - identical
        result = interceptor("test:dataset")

        assert isinstance(result, pd.DataFrame)

    def test_appended_data_allowed(self, config_for_interceptor: SentinelConfig):
        """Verify appended rows are allowed in append_only mode."""
        call_count = 0

        def mock_get(dataset, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame({"a": [1, 2, 3]})
            else:
                return pd.DataFrame({"a": [1, 2, 3, 4]})  # Added row

        mock_fn = MagicMock(side_effect=mock_get)
        interceptor = DataInterceptor(mock_fn, config_for_interceptor)

        # First call
        interceptor("test:dataset")

        # Second call with appended data
        result = interceptor("test:dataset")

        assert len(result) == 4

    def test_deleted_data_raises_error(self, config_for_interceptor: SentinelConfig):
        """Verify deleted rows raise error in append_only mode."""
        call_count = 0

        def mock_get(dataset, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
            else:
                return pd.DataFrame({"a": [1, 2]}, index=[0, 1])  # Deleted row

        mock_fn = MagicMock(side_effect=mock_get)
        interceptor = DataInterceptor(mock_fn, config_for_interceptor)

        # First call
        interceptor("test:dataset")

        # Second call with deleted data - should raise
        with pytest.raises(DataAnomalyError):
            interceptor("test:dataset")

    def test_warn_return_cached_behavior(self, tmp_path: Path):
        """Verify warn_return_cached returns cached data."""
        config = SentinelConfig(
            storage=StorageConfig(path=tmp_path),
            anomaly=AnomalyConfig(behavior=AnomalyBehavior.WARN_RETURN_CACHED),
        )

        call_count = 0

        def mock_get(dataset, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
            else:
                return pd.DataFrame({"a": [1, 2]}, index=[0, 1])

        mock_fn = MagicMock(side_effect=mock_get)
        interceptor = DataInterceptor(mock_fn, config)

        # First call
        first_result = interceptor("test:dataset")

        # Second call - should warn and return cached
        with pytest.warns(UserWarning):
            result = interceptor("test:dataset")

        # Should return cached (3 rows)
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, first_result)

    def test_warn_return_new_behavior(self, tmp_path: Path):
        """Verify warn_return_new returns new data."""
        config = SentinelConfig(
            storage=StorageConfig(path=tmp_path),
            anomaly=AnomalyConfig(behavior=AnomalyBehavior.WARN_RETURN_NEW),
        )

        call_count = 0

        def mock_get(dataset, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
            else:
                return pd.DataFrame({"a": [1, 2]}, index=[0, 1])

        mock_fn = MagicMock(side_effect=mock_get)
        interceptor = DataInterceptor(mock_fn, config)

        # First call
        interceptor("test:dataset")

        # Second call - should warn and return new
        with pytest.warns(UserWarning):
            result = interceptor("test:dataset")

        # Should return new (2 rows)
        assert len(result) == 2

    def test_converts_non_dataframe_result(
        self, config_for_interceptor: SentinelConfig
    ):
        """Verify non-DataFrame results are converted."""
        # Return something that looks like a DataFrame but isn't exactly pd.DataFrame
        mock_fn = MagicMock(return_value={"a": [1, 2, 3], "b": [4, 5, 6]})

        interceptor = DataInterceptor(mock_fn, config_for_interceptor)

        result = interceptor("test:dataset")

        assert isinstance(result, pd.DataFrame)

    def test_original_get_failure_propagates(
        self, config_for_interceptor: SentinelConfig
    ):
        """Verify original get failures are propagated."""
        mock_fn = MagicMock(side_effect=RuntimeError("API Error"))

        interceptor = DataInterceptor(mock_fn, config_for_interceptor)

        with pytest.raises(RuntimeError, match="API Error"):
            interceptor("test:dataset")

    def test_history_modifiable_dataset_allowed(self, tmp_path: Path):
        """Verify datasets in history_modifiable list allow modifications."""
        config = SentinelConfig(
            storage=StorageConfig(path=tmp_path),
            comparison=ComparisonConfig(
                policies=ComparisonPoliciesConfig(
                    default_mode=PolicyMode.APPEND_ONLY,
                    history_modifiable=["test:modifiable"],
                )
            ),
            anomaly=AnomalyConfig(behavior=AnomalyBehavior.RAISE),
        )

        call_count = 0

        def mock_get(dataset, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
            else:
                return pd.DataFrame({"a": [1, 2]}, index=[0, 1])  # Deleted row

        mock_fn = MagicMock(side_effect=mock_get)
        interceptor = DataInterceptor(mock_fn, config)

        # First call
        interceptor("test:modifiable")

        # Second call - should NOT raise because it's in history_modifiable
        result = interceptor("test:modifiable")
        assert len(result) == 2

    def test_report_saved_when_configured(self, tmp_path: Path):
        """Verify anomaly reports are saved when configured."""
        config = SentinelConfig(
            storage=StorageConfig(path=tmp_path),
            anomaly=AnomalyConfig(
                behavior=AnomalyBehavior.WARN_RETURN_NEW, save_reports=True
            ),
        )

        call_count = 0

        def mock_get(dataset, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
            else:
                return pd.DataFrame({"a": [1, 2]}, index=[0, 1])

        mock_fn = MagicMock(side_effect=mock_get)
        interceptor = DataInterceptor(mock_fn, config)

        interceptor("test:dataset")

        with pytest.warns(UserWarning):
            interceptor("test:dataset")

        # Check if report was saved
        reports_path = config.get_reports_path()
        report_files = list(reports_path.glob("*.json"))
        assert len(report_files) >= 1


class TestPreprocessHookInInterceptor:
    """Tests for preprocess hook integration with interceptor."""

    def test_storage_contains_raw_data_not_preprocessed(self, tmp_path: Path):
        """Verify storage saves raw data, not preprocessed data."""
        from finlab_sentinel.core.hooks import (
            clear_preprocess_hooks,
            register_preprocess_hook,
        )

        config = SentinelConfig(
            storage=StorageConfig(path=tmp_path),
            anomaly=AnomalyConfig(behavior=AnomalyBehavior.RAISE),
        )

        # Register a preprocess hook that rounds to 0 decimal places
        register_preprocess_hook("test:dataset", lambda df: df.round(0))

        try:
            # Return data with decimal values
            mock_fn = MagicMock(return_value=pd.DataFrame({"a": [1.234, 2.567, 3.891]}))
            interceptor = DataInterceptor(mock_fn, config)

            # First call saves baseline
            result = interceptor("test:dataset")

            # Result should be original (not rounded)
            assert result["a"].iloc[0] == 1.234

            # Stored data should also be original (not rounded)
            cached = interceptor.storage.load_latest("test__dataset")
            assert cached is not None
            cached_df, _ = cached
            assert cached_df["a"].iloc[0] == 1.234  # Raw data, not rounded

        finally:
            clear_preprocess_hooks()

    def test_warn_return_cached_returns_raw_data_with_preprocess_hook(
        self, tmp_path: Path
    ):
        """Verify warn_return_cached returns raw cached data, not preprocessed."""
        from finlab_sentinel.core.hooks import (
            clear_preprocess_hooks,
            register_preprocess_hook,
        )

        config = SentinelConfig(
            storage=StorageConfig(path=tmp_path),
            anomaly=AnomalyConfig(behavior=AnomalyBehavior.WARN_RETURN_CACHED),
        )

        # Register a preprocess hook that rounds to 0 decimal places
        register_preprocess_hook("test:dataset", lambda df: df.round(0))

        call_count = 0

        def mock_get(dataset, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame({"a": [1.234, 2.567, 3.891]}, index=[0, 1, 2])
            else:
                # Delete a row to trigger anomaly
                return pd.DataFrame({"a": [1.5, 2.5]}, index=[0, 1])

        try:
            mock_fn = MagicMock(side_effect=mock_get)
            interceptor = DataInterceptor(mock_fn, config)

            # First call - saves raw data
            first_result = interceptor("test:dataset")
            assert first_result["a"].iloc[0] == 1.234

            # Second call - should warn and return cached RAW data
            with pytest.warns(UserWarning):
                result = interceptor("test:dataset")

            # Returned cached data should be raw (1.234), not preprocessed (1.0)
            assert len(result) == 3
            assert result["a"].iloc[0] == 1.234

        finally:
            clear_preprocess_hooks()


class TestAcceptCurrentData:
    """Tests for accept_current_data function."""

    def test_accept_updates_baseline(self, config_for_interceptor: SentinelConfig):
        """Verify accepting data updates the baseline."""
        # Create a mock finlab module
        import sys
        from types import ModuleType

        from finlab_sentinel.storage.parquet import ParquetStorage, sanitize_backup_key

        # First, directly save some data to storage
        storage = ParquetStorage(
            base_path=config_for_interceptor.get_storage_path(),
            compression=config_for_interceptor.storage.compression,
        )
        test_df = pd.DataFrame({"a": [1, 2, 3]})
        backup_key = sanitize_backup_key("test:dataset")
        storage.save(backup_key, "test:dataset", test_df, "old_hash")

        mock_data = MagicMock()
        mock_data.get = MagicMock(return_value=pd.DataFrame({"a": [4, 5, 6]}))

        mock_finlab = ModuleType("finlab")
        mock_finlab.data = mock_data

        sys.modules["finlab"] = mock_finlab

        try:
            # Accept current data
            result = accept_current_data(
                "test:dataset", config_for_interceptor, "test reason"
            )

            assert result is True
        finally:
            if "finlab" in sys.modules:
                del sys.modules["finlab"]

    def test_accept_returns_false_for_unknown_dataset(
        self, config_for_interceptor: SentinelConfig
    ):
        """Verify returns False for unknown dataset."""
        import sys
        from types import ModuleType

        mock_data = MagicMock()
        mock_data.get = MagicMock(return_value=pd.DataFrame({"a": [1]}))

        mock_finlab = ModuleType("finlab")
        mock_finlab.data = mock_data

        sys.modules["finlab"] = mock_finlab

        try:
            result = accept_current_data("unknown:dataset", config_for_interceptor)
            assert result is False
        finally:
            if "finlab" in sys.modules:
                del sys.modules["finlab"]

    def test_accept_with_default_config(self, tmp_path: Path):
        """Verify accept_current_data loads default config when None."""
        import sys
        from types import ModuleType
        from unittest.mock import patch

        from finlab_sentinel.storage.parquet import ParquetStorage, sanitize_backup_key

        # Create a config to use
        config = SentinelConfig(storage=StorageConfig(path=tmp_path))

        # First, directly save some data to storage
        storage = ParquetStorage(
            base_path=config.get_storage_path(),
            compression=config.storage.compression,
        )
        test_df = pd.DataFrame({"a": [1, 2, 3]})
        backup_key = sanitize_backup_key("test:dataset")
        storage.save(backup_key, "test:dataset", test_df, "old_hash")

        mock_data = MagicMock()
        mock_data.get = MagicMock(return_value=pd.DataFrame({"a": [4, 5, 6]}))

        mock_finlab = ModuleType("finlab")
        mock_finlab.data = mock_data

        sys.modules["finlab"] = mock_finlab

        try:
            # Patch load_config in the config.loader module (where it's imported from)
            with patch(
                "finlab_sentinel.config.loader.load_config", return_value=config
            ):
                # Call without config parameter
                result = accept_current_data("test:dataset")

            assert result is True
        finally:
            if "finlab" in sys.modules:
                del sys.modules["finlab"]

    def test_accept_returns_false_on_fetch_error(
        self, config_for_interceptor: SentinelConfig
    ):
        """Verify accept returns False when fetching data fails."""
        import sys
        from types import ModuleType

        from finlab_sentinel.storage.parquet import ParquetStorage, sanitize_backup_key

        # First, directly save some data to storage
        storage = ParquetStorage(
            base_path=config_for_interceptor.get_storage_path(),
            compression=config_for_interceptor.storage.compression,
        )
        test_df = pd.DataFrame({"a": [1, 2, 3]})
        backup_key = sanitize_backup_key("test:dataset")
        storage.save(backup_key, "test:dataset", test_df, "old_hash")

        # Mock finlab to raise an error
        mock_data = MagicMock()
        mock_data.get = MagicMock(side_effect=RuntimeError("API Error"))

        mock_finlab = ModuleType("finlab")
        mock_finlab.data = mock_data

        sys.modules["finlab"] = mock_finlab

        try:
            result = accept_current_data("test:dataset", config_for_interceptor)
            assert result is False
        finally:
            if "finlab" in sys.modules:
                del sys.modules["finlab"]


class TestUniverseHashDetection:
    """Tests for universe hash detection in interceptor."""

    def test_universe_hash_from_universe_attribute(self, tmp_path: Path):
        """Verify universe hash is detected from _universe attribute."""
        import sys
        from types import ModuleType

        config = SentinelConfig(
            storage=StorageConfig(path=tmp_path),
            anomaly=AnomalyConfig(behavior=AnomalyBehavior.RAISE),
        )

        # Create mock finlab with _universe attribute
        mock_data = MagicMock()
        mock_data._universe = "SP500"
        mock_data.get = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3]}))

        mock_finlab = ModuleType("finlab")
        mock_finlab.data = mock_data

        sys.modules["finlab"] = mock_finlab

        try:
            mock_fn = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3]}))
            interceptor = DataInterceptor(mock_fn, config)

            # Call to trigger universe hash detection
            interceptor("test:dataset")

            # Backup key should include universe hash
            backups = interceptor.storage.list_backups()
            assert len(backups) == 1
            # Key should contain universe hash
            assert "universe" in backups[0].backup_key

        finally:
            if "finlab" in sys.modules:
                del sys.modules["finlab"]

    def test_no_universe_hash_when_not_set(
        self, config_for_interceptor: SentinelConfig
    ):
        """Verify backup key doesn't include universe hash when not set."""
        mock_fn = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3]}))
        interceptor = DataInterceptor(mock_fn, config_for_interceptor)

        # Call to save backup
        interceptor("test:dataset")

        # Check backup key doesn't include universe
        backups = interceptor.storage.list_backups()
        assert len(backups) == 1
        assert "universe" not in backups[0].backup_key

    def test_universe_hash_from_universe_context(self, tmp_path: Path):
        """Verify universe hash is detected from universe context."""
        import sys
        from types import ModuleType

        config = SentinelConfig(
            storage=StorageConfig(path=tmp_path),
            anomaly=AnomalyConfig(behavior=AnomalyBehavior.RAISE),
        )

        # Create mock finlab with universe context
        mock_universe = MagicMock()
        mock_universe._current = "NASDAQ100"

        mock_data = MagicMock()
        mock_data._universe = None  # No _universe attribute
        mock_data.universe = mock_universe
        mock_data.get = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3]}))

        mock_finlab = ModuleType("finlab")
        mock_finlab.data = mock_data

        sys.modules["finlab"] = mock_finlab

        try:
            mock_fn = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3]}))
            interceptor = DataInterceptor(mock_fn, config)

            # Call to trigger universe hash detection
            interceptor("test:dataset")

            # Backup key should include universe hash
            backups = interceptor.storage.list_backups()
            assert len(backups) == 1
            assert "universe" in backups[0].backup_key

        finally:
            if "finlab" in sys.modules:
                del sys.modules["finlab"]


class TestHashMismatchButIdentical:
    """Tests for edge case where hash mismatches but content is identical."""

    def test_hash_mismatch_but_identical_content(self, tmp_path: Path):
        """Verify handling when hash mismatches but full comparison shows identical."""
        from unittest.mock import patch

        config = SentinelConfig(
            storage=StorageConfig(path=tmp_path),
            anomaly=AnomalyConfig(behavior=AnomalyBehavior.RAISE),
        )

        call_count = 0
        test_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

        def mock_get(dataset, *args, **kwargs):
            return test_df.copy()

        mock_fn = MagicMock(side_effect=mock_get)
        interceptor = DataInterceptor(mock_fn, config)

        # First call to establish baseline
        interceptor("test:dataset")

        # Patch hasher to return different hash for same data
        original_hash = interceptor.hasher.hash_dataframe

        def fake_hash(df):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return original_hash(df)
            return "different_hash_" + str(call_count)

        with patch.object(interceptor.hasher, "hash_dataframe", side_effect=fake_hash):
            # Second call - hash will mismatch but content is identical
            result = interceptor("test:dataset")

        # Should return data without error
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
