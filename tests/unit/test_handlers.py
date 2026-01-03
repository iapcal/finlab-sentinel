"""Tests for anomaly handlers."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from finlab_sentinel.comparison.differ import ComparisonResult
from finlab_sentinel.comparison.report import AnomalyReport
from finlab_sentinel.exceptions import DataAnomalyError
from finlab_sentinel.handlers.callback import (
    CallbackHandler,
    create_handler_from_config,
)
from finlab_sentinel.handlers.exception import RaiseExceptionHandler
from finlab_sentinel.handlers.warning import (
    WarnReturnCachedHandler,
    WarnReturnNewHandler,
)


@pytest.fixture
def sample_report() -> AnomalyReport:
    """Create sample anomaly report."""
    return AnomalyReport(
        dataset="price:收盤價",
        backup_key="price__收盤價",
        detected_at=datetime.now(),
        comparison_result=ComparisonResult(
            is_identical=False,
            deleted_rows={"2025-01-01"},
            old_shape=(10, 4),
            new_shape=(9, 4),
        ),
        policy_name="append_only",
        violation_message="Deleted rows detected",
    )


class TestRaiseExceptionHandler:
    """Tests for RaiseExceptionHandler."""

    def test_raises_data_anomaly_error(
        self, sample_report: AnomalyReport, sample_df: pd.DataFrame
    ):
        """Verify exception is raised with report."""
        handler = RaiseExceptionHandler()

        with pytest.raises(DataAnomalyError) as exc_info:
            handler.handle(sample_report, sample_df, sample_df)

        assert exc_info.value.report == sample_report


class TestWarnReturnCachedHandler:
    """Tests for WarnReturnCachedHandler."""

    def test_returns_cached_data(
        self, sample_report: AnomalyReport, sample_df: pd.DataFrame
    ):
        """Verify cached data is returned."""
        handler = WarnReturnCachedHandler()
        cached = sample_df.copy()
        new = sample_df.copy()
        new.iloc[0, 0] = 999999

        with pytest.warns(UserWarning):
            result = handler.handle(sample_report, cached, new)

        pd.testing.assert_frame_equal(result, cached)


class TestWarnReturnNewHandler:
    """Tests for WarnReturnNewHandler."""

    def test_returns_new_data(
        self, sample_report: AnomalyReport, sample_df: pd.DataFrame
    ):
        """Verify new data is returned."""
        handler = WarnReturnNewHandler()
        cached = sample_df.copy()
        new = sample_df.copy()
        new.iloc[0, 0] = 999999

        with pytest.warns(UserWarning):
            result = handler.handle(sample_report, cached, new)

        pd.testing.assert_frame_equal(result, new)


class TestCallbackHandler:
    """Tests for CallbackHandler."""

    def test_callback_is_invoked(
        self, sample_report: AnomalyReport, sample_df: pd.DataFrame
    ):
        """Verify callback function is called."""
        callback = MagicMock()
        fallback = WarnReturnNewHandler()
        handler = CallbackHandler(callback, fallback)

        with pytest.warns(UserWarning):
            handler.handle(sample_report, sample_df, sample_df)

        callback.assert_called_once_with(sample_report)

    def test_fallback_determines_return(
        self, sample_report: AnomalyReport, sample_df: pd.DataFrame
    ):
        """Verify fallback handler determines return value."""
        callback = MagicMock()
        fallback = WarnReturnCachedHandler()
        handler = CallbackHandler(callback, fallback)

        cached = sample_df.copy()
        new = sample_df.copy()
        new.iloc[0, 0] = 999999

        with pytest.warns(UserWarning):
            result = handler.handle(sample_report, cached, new)

        # Should return cached (from fallback)
        pd.testing.assert_frame_equal(result, cached)

    def test_callback_exception_handled(
        self, sample_report: AnomalyReport, sample_df: pd.DataFrame
    ):
        """Verify callback exceptions don't crash handler."""

        def bad_callback(report):
            raise RuntimeError("Callback failed")

        fallback = WarnReturnNewHandler()
        handler = CallbackHandler(bad_callback, fallback)

        # Should not raise, but log error
        with pytest.warns(UserWarning):
            result = handler.handle(sample_report, sample_df, sample_df)

        # Should still return data from fallback
        pd.testing.assert_frame_equal(result, sample_df)


class TestCreateHandlerFromConfig:
    """Tests for create_handler_from_config function."""

    def test_raise_behavior(self):
        """Verify raise behavior creates RaiseExceptionHandler."""
        handler = create_handler_from_config("raise")
        assert isinstance(handler, RaiseExceptionHandler)

    def test_warn_return_cached_behavior(self):
        """Verify warn_return_cached behavior."""
        handler = create_handler_from_config("warn_return_cached")
        assert isinstance(handler, WarnReturnCachedHandler)

    def test_warn_return_new_behavior(self):
        """Verify warn_return_new behavior."""
        handler = create_handler_from_config("warn_return_new")
        assert isinstance(handler, WarnReturnNewHandler)

    def test_with_callback(self):
        """Verify callback wrapping."""
        callback = MagicMock()
        handler = create_handler_from_config("warn_return_new", callback)

        assert isinstance(handler, CallbackHandler)

    def test_unknown_behavior_defaults_to_raise(self):
        """Verify unknown behavior defaults to raise."""
        handler = create_handler_from_config("unknown_behavior")
        assert isinstance(handler, RaiseExceptionHandler)
