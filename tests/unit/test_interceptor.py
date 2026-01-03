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
