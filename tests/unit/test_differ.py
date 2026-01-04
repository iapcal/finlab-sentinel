"""Tests for DataFrame comparer."""

import numpy as np
import pandas as pd

from finlab_sentinel.comparison.differ import DataFrameComparer


class TestDataFrameComparer:
    """Tests for DataFrameComparer class."""

    def test_identical_dataframes(self, sample_df: pd.DataFrame):
        """Verify identical DataFrames show no changes."""
        comparer = DataFrameComparer()

        result = comparer.compare(sample_df, sample_df.copy())

        assert result.is_identical
        assert result.total_changes == 0
        assert result.change_ratio == 0.0

    def test_detect_added_rows(
        self, sample_df: pd.DataFrame, sample_df_appended: pd.DataFrame
    ):
        """Verify new rows are detected."""
        comparer = DataFrameComparer()

        result = comparer.compare(sample_df, sample_df_appended)

        assert len(result.added_rows) == 3
        assert len(result.deleted_rows) == 0
        assert result.is_append_only()

    def test_detect_deleted_rows(
        self, sample_df: pd.DataFrame, sample_df_appended: pd.DataFrame
    ):
        """Verify deleted rows are detected."""
        comparer = DataFrameComparer()

        result = comparer.compare(sample_df_appended, sample_df)

        assert len(result.deleted_rows) == 3
        assert len(result.added_rows) == 0
        assert not result.is_append_only()

    def test_detect_modified_values(
        self, sample_df: pd.DataFrame, sample_df_modified: pd.DataFrame
    ):
        """Verify value modifications are detected."""
        comparer = DataFrameComparer()

        result = comparer.compare(sample_df, sample_df_modified)

        assert len(result.modified_cells) == 2
        assert not result.is_append_only()

    def test_tolerance_respected(self, sample_df: pd.DataFrame):
        """Verify values within tolerance are not flagged."""
        comparer = DataFrameComparer(rtol=1e-5, atol=1e-8)

        df_modified = sample_df.copy()
        # Add tiny change within tolerance
        df_modified.iloc[0, 0] = sample_df.iloc[0, 0] + 1e-10

        result = comparer.compare(sample_df, df_modified)

        assert len(result.modified_cells) == 0

    def test_detect_dtype_changes(self, sample_df: pd.DataFrame):
        """Verify dtype changes are detected."""
        comparer = DataFrameComparer(check_dtype=True)

        df_modified = sample_df.copy()
        df_modified["2330"] = df_modified["2330"].astype("float32")

        result = comparer.compare(sample_df, df_modified)

        assert len(result.dtype_changes) == 1
        assert result.dtype_changes[0].column == "2330"

    def test_detect_na_type_differences(self):
        """Verify pd.NA vs np.nan vs None differences are detected."""
        comparer = DataFrameComparer(check_na_type=True)

        dates = pd.date_range("2025-01-01", periods=3)
        # Use object dtype to preserve actual NA types
        old_arr = pd.array([np.nan, "b", "c"], dtype=object)
        new_arr = pd.array([pd.NA, "b", "c"], dtype=object)
        old_df = pd.DataFrame({"col": old_arr}, index=dates)
        new_df = pd.DataFrame({"col": new_arr}, index=dates)

        result = comparer.compare(old_df, new_df)

        assert len(result.na_type_changes) == 1

    def test_change_ratio_calculation(self, sample_df: pd.DataFrame):
        """Verify change ratio is calculated correctly."""
        comparer = DataFrameComparer()

        # Delete half the rows
        df_modified = sample_df.iloc[:5]

        result = comparer.compare(sample_df, df_modified)

        # 5 deleted rows × 4 columns = 20 cells deleted
        # Original: 10 × 4 = 40 cells
        # Expected ratio: 20 / 40 = 0.5
        assert 0.4 < result.change_ratio < 0.6

    def test_added_columns(self, sample_df: pd.DataFrame):
        """Verify new columns are detected."""
        comparer = DataFrameComparer()

        df_modified = sample_df.copy()
        df_modified["NEW_COL"] = 100.0

        result = comparer.compare(sample_df, df_modified)

        assert "NEW_COL" in result.added_columns
        assert result.is_append_only()

    def test_deleted_columns(self, sample_df: pd.DataFrame):
        """Verify deleted columns are detected."""
        comparer = DataFrameComparer()

        df_modified = sample_df.drop(columns=["2330"])

        result = comparer.compare(sample_df, df_modified)

        assert "2330" in result.deleted_columns
        assert not result.is_append_only()

    def test_summary_output(
        self, sample_df: pd.DataFrame, sample_df_modified: pd.DataFrame
    ):
        """Verify summary is readable."""
        comparer = DataFrameComparer()

        result = comparer.compare(sample_df, sample_df_modified)
        summary = result.summary()

        assert "modified cells" in summary
        assert "change ratio" in summary

    def test_empty_dataframes(self):
        """Verify empty DataFrames comparison works."""
        comparer = DataFrameComparer()

        empty1 = pd.DataFrame()
        empty2 = pd.DataFrame()

        result = comparer.compare(empty1, empty2)

        assert result.is_identical

    def test_detect_na_to_value_changes(self):
        """Verify NA→value transitions are tracked separately."""
        comparer = DataFrameComparer()

        dates = pd.date_range("2025-01-01", periods=3)
        old_df = pd.DataFrame(
            {"col1": [np.nan, 2.0, 3.0], "col2": [1.0, np.nan, 3.0]},
            index=dates,
        )
        new_df = pd.DataFrame(
            {"col1": [100.0, 2.0, 3.0], "col2": [1.0, 200.0, 3.0]},
            index=dates,
        )

        result = comparer.compare(old_df, new_df)

        # Both NA→value changes should be tracked
        assert result.na_to_value_cells_count == 2
        assert len(result.na_to_value_cells) == 2
        # They're also in modified_cells
        assert result.modified_cells_count == 2

    def test_is_append_only_with_ignore_na_to_value(self):
        """Verify is_append_only respects ignore_na_to_value flag."""
        comparer = DataFrameComparer()

        dates = pd.date_range("2025-01-01", periods=2)
        old_df = pd.DataFrame({"col": [np.nan, 2.0]}, index=dates)
        new_df = pd.DataFrame({"col": [100.0, 2.0]}, index=dates)

        result = comparer.compare(old_df, new_df)

        # Without ignore: not append-only
        assert not result.is_append_only()
        # With ignore: is append-only (only NA→value change)
        assert result.is_append_only(ignore_na_to_value=True)

    def test_value_to_na_not_ignored(self):
        """Verify value→NA changes are NOT ignored by ignore_na_to_value."""
        comparer = DataFrameComparer()

        dates = pd.date_range("2025-01-01", periods=2)
        old_df = pd.DataFrame({"col": [100.0, 2.0]}, index=dates)
        new_df = pd.DataFrame({"col": [np.nan, 2.0]}, index=dates)

        result = comparer.compare(old_df, new_df)

        # Has modified cells but no NA→value
        assert result.modified_cells_count == 1
        assert result.na_to_value_cells_count == 0
        # Still not append-only even with ignore
        assert not result.is_append_only(ignore_na_to_value=True)
