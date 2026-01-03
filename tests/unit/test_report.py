"""Tests for anomaly report generation and storage."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finlab_sentinel.comparison.differ import (
    CellChange,
    ChangeType,
    ComparisonResult,
    DtypeChange,
)
from finlab_sentinel.comparison.report import (
    AnomalyReport,
    _serialize_value,
    list_reports,
)


@pytest.fixture
def sample_comparison_result() -> ComparisonResult:
    """Create sample comparison result for testing."""
    return ComparisonResult(
        is_identical=False,
        old_shape=(10, 4),
        new_shape=(9, 4),
        added_rows={"2025-01-11"},
        deleted_rows={"2025-01-01"},
        added_columns=set(),
        deleted_columns=set(),
        modified_cells=[
            CellChange(
                row="2025-01-05",
                column="2330",
                old_value=100.0,
                new_value=101.0,
                change_type=ChangeType.VALUE_MODIFIED,
            )
        ],
        dtype_changes=[DtypeChange("col1", "float64", "float32")],
        na_type_changes=[
            CellChange(
                row="2025-01-02",
                column="2331",
                old_value=np.nan,
                new_value=pd.NA,
                change_type=ChangeType.NA_TYPE_CHANGED,
            )
        ],
    )


@pytest.fixture
def sample_report(sample_comparison_result: ComparisonResult) -> AnomalyReport:
    """Create sample anomaly report."""
    return AnomalyReport(
        dataset="price:收盤價",
        backup_key="price__收盤價",
        detected_at=datetime(2025, 1, 10, 12, 30, 0),
        comparison_result=sample_comparison_result,
        policy_name="append_only",
        violation_message="Deleted rows detected",
        old_hash="abc123",
        new_hash="def456",
    )


class TestAnomalyReport:
    """Tests for AnomalyReport class."""

    def test_summary_property(self, sample_report: AnomalyReport):
        """Verify summary includes key info."""
        summary = sample_report.summary

        assert "price:收盤價" in summary
        assert "append_only violated" in summary

    def test_to_dict(self, sample_report: AnomalyReport):
        """Verify to_dict serialization."""
        data = sample_report.to_dict()

        assert data["dataset"] == "price:收盤價"
        assert data["backup_key"] == "price__收盤價"
        assert data["policy_name"] == "append_only"
        assert data["old_hash"] == "abc123"
        assert data["new_hash"] == "def456"
        assert data["comparison"]["is_identical"] is False
        assert data["comparison"]["old_shape"] == [10, 4]
        assert len(data["comparison"]["modified_cells"]) == 1
        assert len(data["comparison"]["dtype_changes"]) == 1
        assert len(data["comparison"]["na_type_changes"]) == 1

    def test_to_json(self, sample_report: AnomalyReport):
        """Verify JSON serialization."""
        json_str = sample_report.to_json()

        assert isinstance(json_str, str)
        assert "price:收盤價" in json_str
        assert "append_only" in json_str

    def test_save_and_load(self, sample_report: AnomalyReport, tmp_path: Path):
        """Verify save and load roundtrip."""
        # Save
        filepath = sample_report.save(tmp_path)

        assert filepath.exists()
        assert "anomaly.json" in filepath.name

        # Load
        loaded = AnomalyReport.load(filepath)

        assert loaded.dataset == sample_report.dataset
        assert loaded.backup_key == sample_report.backup_key
        assert loaded.policy_name == sample_report.policy_name
        assert loaded.old_hash == sample_report.old_hash

    def test_from_dict(self, sample_report: AnomalyReport):
        """Verify from_dict deserialization."""
        data = sample_report.to_dict()
        loaded = AnomalyReport.from_dict(data)

        assert loaded.dataset == sample_report.dataset
        assert (
            loaded.comparison_result.is_identical
            == sample_report.comparison_result.is_identical
        )
        assert len(loaded.comparison_result.modified_cells) == len(
            sample_report.comparison_result.modified_cells
        )

    def test_save_creates_directory(self, sample_report: AnomalyReport, tmp_path: Path):
        """Verify save creates directory if needed."""
        nested_path = tmp_path / "nested" / "reports"

        filepath = sample_report.save(nested_path)

        assert filepath.exists()
        assert nested_path.exists()


class TestListReports:
    """Tests for list_reports function."""

    def test_empty_directory(self, tmp_path: Path):
        """Verify empty list for empty directory."""
        reports = list_reports(tmp_path)
        assert reports == []

    def test_nonexistent_directory(self, tmp_path: Path):
        """Verify empty list for nonexistent directory."""
        nonexistent = tmp_path / "does_not_exist"
        reports = list_reports(nonexistent)
        assert reports == []

    def test_lists_reports(self, sample_report: AnomalyReport, tmp_path: Path):
        """Verify reports are listed."""
        # Save multiple reports
        sample_report.save(tmp_path)

        # Change timestamp and save again
        sample_report.detected_at = datetime(2025, 1, 11, 12, 0, 0)
        sample_report.save(tmp_path)

        reports = list_reports(tmp_path)

        assert len(reports) == 2
        # Should be sorted newest first
        assert "2025-01-11" in reports[0].name


class TestSerializeValue:
    """Tests for _serialize_value function."""

    def test_serialize_none(self):
        """Verify None serialization."""
        result = _serialize_value(None)
        assert result == {"__na__": "None"}

    def test_serialize_pd_na(self):
        """Verify pd.NA serialization."""
        result = _serialize_value(pd.NA)
        assert result == {"__na__": "pd.NA"}

    def test_serialize_np_nan(self):
        """Verify np.nan serialization."""
        result = _serialize_value(np.nan)
        assert result == {"__na__": "np.nan"}

    def test_serialize_numpy_int(self):
        """Verify numpy int serialization."""
        result = _serialize_value(np.int64(42))
        assert result == 42.0

    def test_serialize_numpy_float(self):
        """Verify numpy float serialization."""
        result = _serialize_value(np.float64(3.14))
        assert result == 3.14

    def test_serialize_numpy_array(self):
        """Verify numpy array serialization."""
        result = _serialize_value(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_serialize_datetime(self):
        """Verify datetime serialization."""
        dt = datetime(2025, 1, 10, 12, 30, 0)
        result = _serialize_value(dt)
        assert result == "2025-01-10T12:30:00"

    def test_serialize_timestamp(self):
        """Verify pandas Timestamp serialization."""
        ts = pd.Timestamp("2025-01-10 12:30:00")
        result = _serialize_value(ts)
        assert "2025-01-10" in result

    def test_serialize_string(self):
        """Verify string serialization."""
        result = _serialize_value("test")
        assert result == "test"

    def test_serialize_int(self):
        """Verify int serialization."""
        result = _serialize_value(42)
        assert result == "42"
