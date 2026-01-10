"""Tests for time travel functionality."""

import time
from datetime import datetime, timedelta
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
from finlab_sentinel.core.interceptor import DataInterceptor
from finlab_sentinel.core.time_travel import TimeTravelContext
from finlab_sentinel.exceptions import NoHistoricalDataError


class TestTimeTravelContext:
    """Tests for TimeTravelContext class."""

    def setup_method(self):
        """Reset singleton instance before each test."""
        TimeTravelContext._reset_instance()

    def teardown_method(self):
        """Clean up singleton instance after each test."""
        TimeTravelContext._reset_instance()

    def test_singleton_pattern(self):
        """Verify TimeTravelContext follows singleton pattern."""
        ctx1 = TimeTravelContext.get_instance()
        ctx2 = TimeTravelContext.get_instance()

        assert ctx1 is ctx2

    def test_initial_state_inactive(self):
        """Verify initial state is inactive."""
        ctx = TimeTravelContext.get_instance()

        assert not ctx.is_active()
        assert ctx.target_time is None

    def test_set_target_time(self):
        """Verify target time can be set."""
        ctx = TimeTravelContext.get_instance()
        target = datetime(2024, 1, 5, 14, 30)

        ctx.set_target_time(target)

        assert ctx.is_active()
        assert ctx.target_time == target

    def test_clear_resets_state(self):
        """Verify clear resets to inactive state."""
        ctx = TimeTravelContext.get_instance()
        target = datetime(2024, 1, 5, 14, 30)

        ctx.set_target_time(target)
        ctx.clear()

        assert not ctx.is_active()
        assert ctx.target_time is None

    def test_multiple_set_overrides(self):
        """Verify setting target time multiple times overrides previous."""
        ctx = TimeTravelContext.get_instance()
        target1 = datetime(2024, 1, 5, 14, 30)
        target2 = datetime(2024, 2, 10, 10, 0)

        ctx.set_target_time(target1)
        assert ctx.target_time == target1

        ctx.set_target_time(target2)
        assert ctx.target_time == target2

    def test_is_active_after_set(self):
        """Verify is_active returns True after setting target time."""
        ctx = TimeTravelContext.get_instance()

        assert not ctx.is_active()

        ctx.set_target_time(datetime(2024, 1, 1))

        assert ctx.is_active()

    def test_is_active_after_clear(self):
        """Verify is_active returns False after clearing."""
        ctx = TimeTravelContext.get_instance()
        ctx.set_target_time(datetime(2024, 1, 1))

        assert ctx.is_active()

        ctx.clear()

        assert not ctx.is_active()


@pytest.fixture
def config_for_time_travel(tmp_path: Path) -> SentinelConfig:
    """Create config for time travel tests."""
    return SentinelConfig(
        storage=StorageConfig(path=tmp_path),
        comparison=ComparisonConfig(
            policies=ComparisonPoliciesConfig(default_mode=PolicyMode.THRESHOLD),
            change_threshold=1.0,  # Allow 100% change for tests
        ),
        anomaly=AnomalyConfig(behavior=AnomalyBehavior.RAISE),
    )


class TestDataInterceptorTimeTravel:
    """Tests for DataInterceptor time travel mode."""

    def setup_method(self):
        """Reset time travel context before each test."""
        TimeTravelContext._reset_instance()

    def teardown_method(self):
        """Clean up time travel context after each test."""
        ctx = TimeTravelContext.get_instance()
        ctx.clear()
        TimeTravelContext._reset_instance()

    def test_time_travel_returns_historical_data(
        self, config_for_time_travel: SentinelConfig
    ):
        """Verify time travel mode returns historical backup."""
        # Create mock that returns different data on each call
        call_count = 0

        def mock_get(dataset, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            else:
                return pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})

        mock_fn = MagicMock(side_effect=mock_get)
        interceptor = DataInterceptor(mock_fn, config_for_time_travel)

        # First call - save baseline
        first_result = interceptor("test:dataset")
        base_time = datetime.now() - timedelta(hours=2)

        # Update first backup timestamp to 2 hours ago
        with interceptor.storage.index._connect() as conn:
            conn.execute(
                "UPDATE backups SET created_at = ? WHERE backup_key = ?",
                (base_time.isoformat(), "test__dataset"),
            )

        # Second call - save new data
        time.sleep(1.0)
        _ = interceptor("test:dataset")
        second_time = datetime.now() - timedelta(hours=1)

        # Update second backup timestamp to 1 hour ago
        with interceptor.storage.index._connect() as conn:
            conn.execute(
                """UPDATE backups SET created_at = ?
                   WHERE backup_key = ? AND created_at > ?""",
                (second_time.isoformat(), "test__dataset", base_time.isoformat()),
            )

        # Now activate time travel to time between first and second backup
        ctx = TimeTravelContext.get_instance()
        ctx.set_target_time(base_time + timedelta(minutes=30))

        # Third call should return first data (time travel)
        time_travel_result = interceptor("test:dataset")

        # Verify it returns historical data (first call)
        pd.testing.assert_frame_equal(time_travel_result, first_result)
        assert time_travel_result.iloc[0, 0] == 1
        assert time_travel_result.iloc[0, 0] != 7

    def test_time_travel_raises_error_when_no_backup(
        self, config_for_time_travel: SentinelConfig
    ):
        """Verify time travel raises error when no historical backup exists."""
        mock_fn = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        interceptor = DataInterceptor(mock_fn, config_for_time_travel)

        # Activate time travel without any backups
        ctx = TimeTravelContext.get_instance()
        ctx.set_target_time(datetime.now())

        # Should raise NoHistoricalDataError
        with pytest.raises(NoHistoricalDataError) as exc_info:
            interceptor("test:dataset")

        assert "No backup found" in str(exc_info.value)

    def test_time_travel_raises_error_when_target_before_all_backups(
        self, config_for_time_travel: SentinelConfig
    ):
        """Verify time travel raises error when target time is before all backups."""
        mock_fn = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        interceptor = DataInterceptor(mock_fn, config_for_time_travel)

        # Create backup
        interceptor("test:dataset")

        # Activate time travel to time before backup
        ctx = TimeTravelContext.get_instance()
        ctx.set_target_time(datetime.now() - timedelta(days=1))

        # Should raise NoHistoricalDataError
        with pytest.raises(NoHistoricalDataError) as exc_info:
            interceptor("test:dataset")

        assert "No backup found" in str(exc_info.value)

    def test_time_travel_does_not_call_original_function(
        self, config_for_time_travel: SentinelConfig
    ):
        """Verify time travel mode does not call original data.get function."""
        mock_fn = MagicMock(return_value=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
        interceptor = DataInterceptor(mock_fn, config_for_time_travel)

        # Create backup
        interceptor("test:dataset")
        assert mock_fn.call_count == 1

        # Activate time travel
        ctx = TimeTravelContext.get_instance()
        ctx.set_target_time(datetime.now())

        # Call in time travel mode
        interceptor("test:dataset")

        # Original function should not be called again
        assert mock_fn.call_count == 1

    def test_exit_time_travel_resumes_normal_operation(
        self, config_for_time_travel: SentinelConfig
    ):
        """Verify exiting time travel resumes normal data fetching."""
        call_count = 0

        def mock_get(dataset, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return pd.DataFrame({"a": [call_count], "b": [call_count * 10]})

        mock_fn = MagicMock(side_effect=mock_get)
        interceptor = DataInterceptor(mock_fn, config_for_time_travel)

        # Create first backup
        first_result = interceptor("test:dataset")
        assert first_result.iloc[0, 0] == 1

        # Activate time travel
        ctx = TimeTravelContext.get_instance()
        ctx.set_target_time(datetime.now())

        # Call in time travel mode
        time_travel_result = interceptor("test:dataset")
        assert time_travel_result.iloc[0, 0] == 1

        # Exit time travel
        ctx.clear()

        # Call should now fetch fresh data
        time.sleep(1.0)
        normal_result = interceptor("test:dataset")
        assert normal_result.iloc[0, 0] == 2  # Second call to original function

    def test_time_travel_with_multiple_datasets(
        self, config_for_time_travel: SentinelConfig
    ):
        """Verify time travel works with multiple datasets independently."""
        mock_fn = MagicMock(
            side_effect=lambda ds, *args, **kwargs: pd.DataFrame(
                {"value": [1 if ds == "dataset1" else 2]}
            )
        )
        interceptor = DataInterceptor(mock_fn, config_for_time_travel)

        # Create backups for two datasets
        result1 = interceptor("dataset1")
        result2 = interceptor("dataset2")

        # Activate time travel
        ctx = TimeTravelContext.get_instance()
        ctx.set_target_time(datetime.now())

        # Both datasets should return their historical data
        tt_result1 = interceptor("dataset1")
        tt_result2 = interceptor("dataset2")

        pd.testing.assert_frame_equal(tt_result1, result1)
        pd.testing.assert_frame_equal(tt_result2, result2)


class TestTimeTravelIntegration:
    """Integration tests for complete time travel workflow using public API."""

    def setup_method(self):
        """Reset time travel context before each test."""
        TimeTravelContext._reset_instance()

    def teardown_method(self):
        """Clean up time travel context after each test."""
        import finlab_sentinel

        finlab_sentinel.exit_time_travel()
        TimeTravelContext._reset_instance()

    def test_public_api_set_and_exit_time_travel(
        self, config_for_time_travel: SentinelConfig
    ):
        """Verify public API for setting and exiting time travel mode."""
        import finlab_sentinel

        # Initially inactive
        status = finlab_sentinel.get_time_travel_status()
        assert not status["enabled"]
        assert status["target_time"] is None

        # Set time travel
        target = datetime(2024, 1, 5, 14, 30)
        finlab_sentinel.set_time_travel(target)

        # Check status
        status = finlab_sentinel.get_time_travel_status()
        assert status["enabled"]
        assert status["target_time"] == target.isoformat()

        # Exit time travel
        finlab_sentinel.exit_time_travel()

        # Check status again
        status = finlab_sentinel.get_time_travel_status()
        assert not status["enabled"]
        assert status["target_time"] is None

    def test_full_workflow_with_public_api(
        self, config_for_time_travel: SentinelConfig
    ):
        """Verify complete time travel workflow using public API."""
        import finlab_sentinel

        call_count = 0

        def mock_get(dataset, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return pd.DataFrame({"price": [100, 200, 300]})
            else:
                return pd.DataFrame({"price": [150, 250, 350]})

        mock_fn = MagicMock(side_effect=mock_get)
        interceptor = DataInterceptor(mock_fn, config_for_time_travel)

        # Step 1: Get initial data (creates backup)
        initial_data = interceptor("price:收盤價")
        base_time = datetime.now() - timedelta(hours=2)
        assert initial_data["price"].tolist() == [100, 200, 300]

        # Update first backup timestamp to 2 hours ago
        with interceptor.storage.index._connect() as conn:
            conn.execute(
                "UPDATE backups SET created_at = ? WHERE backup_key = ?",
                (base_time.isoformat(), "price__收盤價"),
            )

        # Step 2: Get new data
        time.sleep(1.0)
        new_data = interceptor("price:收盤價")
        second_time = datetime.now() - timedelta(hours=1)
        assert new_data["price"].tolist() == [150, 250, 350]

        # Update second backup timestamp to 1 hour ago
        with interceptor.storage.index._connect() as conn:
            conn.execute(
                """UPDATE backups SET created_at = ?
                   WHERE backup_key = ? AND created_at > ?""",
                (second_time.isoformat(), "price__收盤價", base_time.isoformat()),
            )

        # Step 3: Set time travel to initial time
        finlab_sentinel.set_time_travel(base_time + timedelta(minutes=30))

        # Verify status
        status = finlab_sentinel.get_time_travel_status()
        assert status["enabled"]

        # Step 4: Get data in time travel mode (should return initial data)
        time_travel_data = interceptor("price:收盤價")
        assert time_travel_data["price"].tolist() == [100, 200, 300]

        # Step 5: Exit time travel
        finlab_sentinel.exit_time_travel()

        # Step 6: Get data again (should call original function)
        time.sleep(1.0)
        _ = interceptor("price:收盤價")
        # This should be the third call, but we accept new data
        # so it will be the latest data

    def test_time_travel_error_handling_with_public_api(
        self, config_for_time_travel: SentinelConfig
    ):
        """Verify error handling when using time travel with no backups."""
        import finlab_sentinel

        mock_fn = MagicMock(return_value=pd.DataFrame({"price": [100, 200, 300]}))
        interceptor = DataInterceptor(mock_fn, config_for_time_travel)

        # Set time travel before creating any backups
        finlab_sentinel.set_time_travel(datetime.now())

        # Should raise error
        with pytest.raises(NoHistoricalDataError) as exc_info:
            interceptor("price:收盤價")

        assert "No backup found" in str(exc_info.value)

        # Clean up
        finlab_sentinel.exit_time_travel()

    def test_time_travel_multiple_time_points(
        self, config_for_time_travel: SentinelConfig
    ):
        """Verify traveling to different time points returns correct data."""
        import finlab_sentinel

        mock_fn = MagicMock(
            side_effect=[
                pd.DataFrame({"value": [1]}),
                pd.DataFrame({"value": [2]}),
                pd.DataFrame({"value": [3]}),
            ]
        )
        interceptor = DataInterceptor(mock_fn, config_for_time_travel)

        base_time = datetime.now() - timedelta(hours=3)

        # Create 3 backups at different times
        _ = interceptor("test")
        time.sleep(1.0)

        # Update timestamps to be 3, 2, and 1 hours ago
        with interceptor.storage.index._connect() as conn:
            # Get all backups and update their timestamps
            rows = conn.execute(
                "SELECT id FROM backups WHERE backup_key = ? ORDER BY created_at ASC",
                ("test",),
            ).fetchall()
            if len(rows) > 0:
                conn.execute(
                    "UPDATE backups SET created_at = ? WHERE id = ?",
                    (base_time.isoformat(), rows[0]["id"]),
                )

        _ = interceptor("test")
        time.sleep(1.0)

        with interceptor.storage.index._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM backups WHERE backup_key = ? ORDER BY created_at ASC",
                ("test",),
            ).fetchall()
            if len(rows) > 1:
                conn.execute(
                    "UPDATE backups SET created_at = ? WHERE id = ?",
                    ((base_time + timedelta(hours=1)).isoformat(), rows[1]["id"]),
                )

        _ = interceptor("test")
        time.sleep(1.0)

        with interceptor.storage.index._connect() as conn:
            rows = conn.execute(
                "SELECT id FROM backups WHERE backup_key = ? ORDER BY created_at ASC",
                ("test",),
            ).fetchall()
            if len(rows) > 2:
                conn.execute(
                    "UPDATE backups SET created_at = ? WHERE id = ?",
                    ((base_time + timedelta(hours=2)).isoformat(), rows[2]["id"]),
                )

        # Travel to time1
        time1 = base_time + timedelta(minutes=30)
        finlab_sentinel.set_time_travel(time1)
        result1 = interceptor("test")
        assert result1["value"].tolist() == [1]

        # Travel to time2
        time2 = base_time + timedelta(hours=1, minutes=30)
        finlab_sentinel.set_time_travel(time2)
        result2 = interceptor("test")
        assert result2["value"].tolist() == [2]

        # Travel to time3
        time3 = base_time + timedelta(hours=2, minutes=30)
        finlab_sentinel.set_time_travel(time3)
        result3 = interceptor("test")
        assert result3["value"].tolist() == [3]

        # Clean up
        finlab_sentinel.exit_time_travel()
