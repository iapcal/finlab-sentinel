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

    def test_compare_float64_nullable_with_pd_na(self):
        """Verify Float64 columns with pd.NA are compared correctly.

        Regression test for: Float64 + pd.NA causes TypeError when DataFrame
        has mixed dtypes (object array from .values preserves pd.NA).
        """
        comparer = DataFrameComparer()

        dates = pd.date_range("2025-01-01", periods=4)

        # Mixed dtypes trigger object array from .values
        old_df = pd.DataFrame(
            {
                "float64_col": [1.0, 2.0, np.nan, 4.0],
                "Float64_col": pd.array([10.0, pd.NA, 30.0, 40.0], dtype="Float64"),
                "str_col": ["a", "b", "c", "d"],
            },
            index=dates,
        )

        new_df = old_df.copy()

        # This should not raise TypeError
        result = comparer.compare(old_df, new_df)

        assert result.is_identical
        assert result.modified_cells_count == 0

    def test_compare_float64_nullable_na_to_value(self):
        """Verify Float64 NA->value transitions are detected."""
        comparer = DataFrameComparer()

        dates = pd.date_range("2025-01-01", periods=3)

        old_df = pd.DataFrame(
            {
                "Float64_col": pd.array([10.0, pd.NA, 30.0], dtype="Float64"),
                "str_col": ["a", "b", "c"],
            },
            index=dates,
        )
        new_df = pd.DataFrame(
            {
                "Float64_col": pd.array([10.0, 20.0, 30.0], dtype="Float64"),
                "str_col": ["a", "b", "c"],
            },
            index=dates,
        )

        result = comparer.compare(old_df, new_df)

        assert not result.is_identical
        assert result.modified_cells_count == 1
        assert result.na_to_value_cells_count == 1

    def test_compare_int64_nullable_with_pd_na(self):
        """Verify Int64 (nullable integer) columns with pd.NA work correctly."""
        comparer = DataFrameComparer()

        dates = pd.date_range("2025-01-01", periods=3)

        old_df = pd.DataFrame(
            {
                "Int64_col": pd.array([1, pd.NA, 3], dtype="Int64"),
                "str_col": ["a", "b", "c"],
            },
            index=dates,
        )

        new_df = old_df.copy()

        result = comparer.compare(old_df, new_df)

        assert result.is_identical

    def test_no_common_cells(self):
        """Verify comparison when there are no common rows/columns."""
        comparer = DataFrameComparer()

        old_df = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        new_df = pd.DataFrame({"b": [3, 4]}, index=[2, 3])

        result = comparer.compare(old_df, new_df)

        # All rows and columns are different
        assert len(result.added_rows) == 2
        assert len(result.deleted_rows) == 2
        assert len(result.added_columns) == 1
        assert len(result.deleted_columns) == 1
        assert not result.is_identical

    def test_summary_no_changes(self):
        """Verify summary for identical data."""
        comparer = DataFrameComparer()

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = comparer.compare(df, df.copy())

        assert result.summary() == "No changes"

    def test_summary_with_added_rows(self):
        """Verify summary includes added rows."""
        comparer = DataFrameComparer()

        old_df = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        new_df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])

        result = comparer.compare(old_df, new_df)

        assert "+1 rows" in result.summary()

    def test_summary_with_deleted_rows(self):
        """Verify summary includes deleted rows."""
        comparer = DataFrameComparer()

        old_df = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
        new_df = pd.DataFrame({"a": [1, 2]}, index=[0, 1])

        result = comparer.compare(old_df, new_df)

        assert "-1 rows" in result.summary()

    def test_summary_with_columns(self):
        """Verify summary includes column changes."""
        comparer = DataFrameComparer()

        old_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        new_df = pd.DataFrame({"a": [1, 2], "c": [5, 6]})

        result = comparer.compare(old_df, new_df)

        assert "+1 columns" in result.summary()
        assert "-1 columns" in result.summary()

    def test_summary_with_dtype_changes(self):
        """Verify summary includes dtype changes."""
        comparer = DataFrameComparer()

        old_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        new_df = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="Int64")})

        result = comparer.compare(old_df, new_df)

        assert "dtype changes" in result.summary()

    def test_change_ratio_with_empty_dfs(self):
        """Verify change_ratio handles empty DataFrames."""
        from finlab_sentinel.comparison.differ import ComparisonResult

        result = ComparisonResult(
            is_identical=True,
            old_shape=(0, 0),
            new_shape=(0, 0),
        )

        assert result.change_ratio == 0.0

    def test_exceeds_threshold(self):
        """Verify exceeds_threshold method."""
        comparer = DataFrameComparer()

        old_df = pd.DataFrame({"a": [1, 2, 3, 4, 5]}, index=range(5))
        new_df = pd.DataFrame({"a": [1, 2]}, index=range(2))  # Delete 3 rows

        result = comparer.compare(old_df, new_df)

        # 60% change (3 out of 5 rows deleted)
        assert result.exceeds_threshold(0.50)
        assert not result.exceeds_threshold(0.80)

    def test_check_dtype_disabled(self):
        """Verify dtype check can be disabled."""
        comparer = DataFrameComparer(check_dtype=False)

        old_df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        new_df = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="Int64")})

        result = comparer.compare(old_df, new_df)

        # No dtype changes should be detected
        assert len(result.dtype_changes) == 0

    def test_check_na_type_disabled(self):
        """Verify NA type check can be disabled."""
        comparer = DataFrameComparer(check_na_type=False)

        dates = pd.date_range("2025-01-01", periods=2)
        old_arr = pd.array([np.nan, "b"], dtype=object)
        new_arr = pd.array([pd.NA, "b"], dtype=object)
        old_df = pd.DataFrame({"col": old_arr}, index=dates)
        new_df = pd.DataFrame({"col": new_arr}, index=dates)

        result = comparer.compare(old_df, new_df)

        # No NA type changes should be detected
        assert len(result.na_type_changes) == 0

    def test_many_modified_cells_count_only(self):
        """Verify large changes only track count, not all details."""
        from finlab_sentinel.comparison.differ import MAX_CELL_CHANGES

        comparer = DataFrameComparer()

        # Create DataFrames with many cell changes
        size = MAX_CELL_CHANGES + 5
        old_df = pd.DataFrame({"a": range(size)})
        new_df = pd.DataFrame({"a": range(100, 100 + size)})  # All different

        result = comparer.compare(old_df, new_df)

        # Count should be accurate
        assert result.modified_cells_count == size
        # But list should not contain all of them
        assert len(result.modified_cells) <= MAX_CELL_CHANGES


class TestCellChange:
    """Tests for CellChange dataclass."""

    def test_str_value_modified(self):
        """Verify string representation for value modification."""
        from finlab_sentinel.comparison.differ import CellChange, ChangeType

        change = CellChange(
            row="2025-01-01",
            column="2330",
            old_value=100.0,
            new_value=200.0,
            change_type=ChangeType.VALUE_MODIFIED,
        )

        result = str(change)
        assert "2025-01-01" in result
        assert "2330" in result
        assert "100.0" in result
        assert "200.0" in result
        assert "->" in result

    def test_str_na_type_changed(self):
        """Verify string representation for NA type change."""
        from finlab_sentinel.comparison.differ import CellChange, ChangeType

        change = CellChange(
            row="2025-01-01",
            column="2330",
            old_value=None,
            new_value=np.nan,
            change_type=ChangeType.NA_TYPE_CHANGED,
        )

        result = str(change)
        assert "NA type changed" in result
        assert "None" in result

    def test_str_other_change_type(self):
        """Verify string representation for other change types."""
        from finlab_sentinel.comparison.differ import CellChange, ChangeType

        change = CellChange(
            row="2025-01-01",
            column="2330",
            old_value=100.0,
            new_value=None,
            change_type=ChangeType.ROW_ADDED,
        )

        result = str(change)
        assert "row_added" in result


class TestDtypeChange:
    """Tests for DtypeChange dataclass."""

    def test_str_representation(self):
        """Verify string representation."""
        from finlab_sentinel.comparison.differ import DtypeChange

        change = DtypeChange(
            column="price",
            old_dtype="float64",
            new_dtype="int64",
        )

        result = str(change)
        assert "price" in result
        assert "float64" in result
        assert "int64" in result
        assert "->" in result


class TestGetNaType:
    """Tests for _get_na_type helper function."""

    def test_get_na_type_none(self):
        """Verify None is identified correctly."""
        from finlab_sentinel.comparison.differ import _get_na_type

        assert _get_na_type(None) == "None"

    def test_get_na_type_pd_na(self):
        """Verify pd.NA is identified correctly."""
        from finlab_sentinel.comparison.differ import _get_na_type

        assert _get_na_type(pd.NA) == "pd.NA"

    def test_get_na_type_np_nan(self):
        """Verify np.nan is identified correctly."""
        from finlab_sentinel.comparison.differ import _get_na_type

        assert _get_na_type(np.nan) == "np.nan"

    def test_get_na_type_not_na(self):
        """Verify non-NA values return 'not_na'."""
        from finlab_sentinel.comparison.differ import _get_na_type

        assert _get_na_type(42) == "not_na"
        assert _get_na_type("hello") == "not_na"
        assert _get_na_type(3.14) == "not_na"
