"""DataFrame difference detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class ChangeType(str, Enum):
    """Types of changes detected."""

    ROW_ADDED = "row_added"
    ROW_DELETED = "row_deleted"
    COLUMN_ADDED = "column_added"
    COLUMN_DELETED = "column_deleted"
    VALUE_MODIFIED = "value_modified"
    DTYPE_CHANGED = "dtype_changed"
    NA_TYPE_CHANGED = "na_type_changed"


@dataclass
class CellChange:
    """Represents a single cell value change."""

    row: Any  # Index value (e.g., date)
    column: Any  # Column name (e.g., stock symbol)
    old_value: Any
    new_value: Any
    change_type: ChangeType

    def __str__(self) -> str:
        if self.change_type == ChangeType.VALUE_MODIFIED:
            return f"[{self.row}, {self.column}]: {self.old_value} -> {self.new_value}"
        elif self.change_type == ChangeType.NA_TYPE_CHANGED:
            return (
                f"[{self.row}, {self.column}]: NA type changed "
                f"{_get_na_type(self.old_value)} -> {_get_na_type(self.new_value)}"
            )
        return f"[{self.row}, {self.column}]: {self.change_type.value}"


@dataclass
class DtypeChange:
    """Represents a dtype change for a column."""

    column: Any
    old_dtype: str
    new_dtype: str

    def __str__(self) -> str:
        return f"{self.column}: {self.old_dtype} -> {self.new_dtype}"


# Maximum number of cell changes to track individually
# Beyond this, only count is reported for performance
MAX_CELL_CHANGES = 10


@dataclass
class ComparisonResult:
    """Result of DataFrame comparison."""

    is_identical: bool
    added_rows: set[Any] = field(default_factory=set)
    deleted_rows: set[Any] = field(default_factory=set)
    added_columns: set[Any] = field(default_factory=set)
    deleted_columns: set[Any] = field(default_factory=set)
    modified_cells: list[CellChange] = field(default_factory=list)
    dtype_changes: list[DtypeChange] = field(default_factory=list)
    na_type_changes: list[CellChange] = field(default_factory=list)
    na_to_value_cells: list[CellChange] = field(default_factory=list)

    # Counts (may be > len(list) when exceeding MAX_CELL_CHANGES)
    modified_cells_count: int = 0
    na_type_changes_count: int = 0
    na_to_value_cells_count: int = 0

    # Metrics
    old_shape: tuple[int, int] = (0, 0)
    new_shape: tuple[int, int] = (0, 0)

    @property
    def total_changes(self) -> int:
        """Total number of changes detected."""
        # Use count fields which may be larger than list lengths
        modified = max(self.modified_cells_count, len(self.modified_cells))
        na_changes = max(self.na_type_changes_count, len(self.na_type_changes))
        return (
            len(self.added_rows)
            + len(self.deleted_rows)
            + len(self.added_columns)
            + len(self.deleted_columns)
            + modified
            + len(self.dtype_changes)
            + na_changes
        )

    @property
    def change_ratio(self) -> float:
        """Calculate ratio of changed cells to total cells."""
        if self.old_shape == (0, 0) and self.new_shape == (0, 0):
            return 0.0

        old_total = self.old_shape[0] * self.old_shape[1]
        new_total = self.new_shape[0] * self.new_shape[1]
        total = max(old_total, new_total, 1)

        # Calculate changes
        # Deleted rows affect all old columns
        deleted_cells = len(self.deleted_rows) * self.old_shape[1]
        # Deleted columns affect all old rows
        deleted_cells += len(self.deleted_columns) * self.old_shape[0]
        # Modified cells - use count fields which may be larger than list lengths
        modified = max(self.modified_cells_count, len(self.modified_cells))
        na_changes = max(self.na_type_changes_count, len(self.na_type_changes))

        return (deleted_cells + modified + na_changes) / total

    def is_append_only(self, ignore_na_to_value: bool = False) -> bool:
        """Check if changes are append-only (no deletions/modifications).

        Args:
            ignore_na_to_value: If True, NA→value changes are allowed

        Returns:
            True if changes are append-only
        """
        modified = max(self.modified_cells_count, len(self.modified_cells))
        na_changes = max(self.na_type_changes_count, len(self.na_type_changes))
        na_to_value = max(self.na_to_value_cells_count, len(self.na_to_value_cells))

        # If ignoring NA→value, subtract from modified count
        if ignore_na_to_value:
            modified = max(0, modified - na_to_value)

        return (
            len(self.deleted_rows) == 0
            and len(self.deleted_columns) == 0
            and modified == 0
            and len(self.dtype_changes) == 0
            and na_changes == 0
        )

    def exceeds_threshold(self, threshold: float) -> bool:
        """Check if change ratio exceeds threshold."""
        return self.change_ratio > threshold

    def summary(self) -> str:
        """Generate human-readable summary."""
        parts = []
        if self.added_rows:
            parts.append(f"+{len(self.added_rows)} rows")
        if self.deleted_rows:
            parts.append(f"-{len(self.deleted_rows)} rows")
        if self.added_columns:
            parts.append(f"+{len(self.added_columns)} columns")
        if self.deleted_columns:
            parts.append(f"-{len(self.deleted_columns)} columns")

        # Use count fields which may be larger than list lengths
        modified = max(self.modified_cells_count, len(self.modified_cells))
        na_changes = max(self.na_type_changes_count, len(self.na_type_changes))

        if modified > 0:
            parts.append(f"{modified} modified cells")
        if self.dtype_changes:
            parts.append(f"{len(self.dtype_changes)} dtype changes")
        if na_changes > 0:
            parts.append(f"{na_changes} NA type changes")

        if not parts:
            return "No changes"

        return f"{', '.join(parts)} ({self.change_ratio:.1%} change ratio)"


class DataFrameComparer:
    """Compares two DataFrames and detects changes."""

    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        check_dtype: bool = True,
        check_na_type: bool = True,
    ) -> None:
        """Initialize comparer.

        Args:
            rtol: Relative tolerance for numeric comparisons
            atol: Absolute tolerance for numeric comparisons
            check_dtype: Whether to check for dtype changes
            check_na_type: Whether to check for NA type differences
        """
        self.rtol = rtol
        self.atol = atol
        self.check_dtype = check_dtype
        self.check_na_type = check_na_type

    def compare(
        self,
        old_df: pd.DataFrame,
        new_df: pd.DataFrame,
    ) -> ComparisonResult:
        """Compare two DataFrames and return detailed differences.

        Uses vectorized numpy operations for performance on large DataFrames.

        Args:
            old_df: Previous/cached DataFrame
            new_df: New DataFrame from data source

        Returns:
            ComparisonResult with all detected changes
        """
        result = ComparisonResult(
            is_identical=True,
            old_shape=(len(old_df), len(old_df.columns)),
            new_shape=(len(new_df), len(new_df.columns)),
        )

        # Compare index (rows)
        old_index = set(old_df.index)
        new_index = set(new_df.index)

        result.added_rows = new_index - old_index
        result.deleted_rows = old_index - new_index

        # Compare columns
        old_columns = set(old_df.columns)
        new_columns = set(new_df.columns)

        result.added_columns = new_columns - old_columns
        result.deleted_columns = old_columns - new_columns

        # Get common rows and columns
        common_idx = old_df.index.intersection(new_df.index)
        common_cols = old_df.columns.intersection(new_df.columns)

        # Check for dtype changes in common columns
        if self.check_dtype:
            for col in common_cols:
                old_dtype = str(old_df[col].dtype)
                new_dtype = str(new_df[col].dtype)
                if old_dtype != new_dtype:
                    result.dtype_changes.append(DtypeChange(col, old_dtype, new_dtype))

        # Early return if no common cells to compare
        if len(common_idx) == 0 or len(common_cols) == 0:
            result.is_identical = result.total_changes == 0
            return result

        # Align DataFrames for vectorized comparison
        old_aligned = old_df.loc[common_idx, common_cols]
        new_aligned = new_df.loc[common_idx, common_cols]

        # Vectorized NA detection
        old_na = pd.isna(old_aligned)
        new_na = pd.isna(new_aligned)

        # Both NA → equal (NA type checked separately)
        both_na = old_na & new_na

        # NA→value: old was NA, new has value
        na_to_value = old_na & ~new_na

        # value→NA: old had value, new is NA (still counts as modified)
        # one_na captures both directions
        one_na = old_na ^ new_na

        # Get numpy arrays for fast comparison
        old_vals = old_aligned.values
        new_vals = new_aligned.values

        # Build comparison mask column by column (handles mixed dtypes)
        equal_mask = np.zeros(old_vals.shape, dtype=bool)

        for col_idx, col in enumerate(common_cols):
            col_dtype = old_aligned[col].dtype

            if pd.api.types.is_numeric_dtype(col_dtype):
                # Numeric: use tolerance comparison
                # Use to_numpy() with na_value to handle nullable types (Float64, Int64)
                # which may contain pd.NA that cannot be converted via astype(float)
                old_col_float = old_aligned[col].to_numpy(dtype=float, na_value=np.nan)
                new_col_float = new_aligned[col].to_numpy(dtype=float, na_value=np.nan)

                with np.errstate(invalid="ignore"):
                    col_equal = np.isclose(
                        old_col_float,
                        new_col_float,
                        rtol=self.rtol,
                        atol=self.atol,
                        equal_nan=True,
                    )
            else:
                # Non-numeric: exact equality (handle object dtype safely)
                old_col = old_vals[:, col_idx]
                new_col = new_vals[:, col_idx]
                col_equal = np.array(
                    [
                        (pd.isna(o) and pd.isna(n)) or (o == n)
                        for o, n in zip(old_col, new_col, strict=True)
                    ]
                )

            equal_mask[:, col_idx] = col_equal

        # Combine: different if one_na OR (not equal AND not both_na)
        # one_na: one is NA, one is not → always different
        # ~equal_mask & ~both_na: values differ and not both NA
        diff_mask = one_na.values | (~equal_mask & ~both_na.values)

        # Find difference positions
        diff_rows, diff_cols = np.where(diff_mask)
        diff_count = len(diff_rows)

        # Find NA→value positions
        na_to_value_rows, na_to_value_cols = np.where(na_to_value.values)
        na_to_value_count = len(na_to_value_rows)

        # Set the counts
        result.modified_cells_count = diff_count
        result.na_to_value_cells_count = na_to_value_count

        # Only create CellChange objects if within limit
        if diff_count > 0 and diff_count <= MAX_CELL_CHANGES:
            for i, j in zip(diff_rows, diff_cols, strict=True):
                row_label = common_idx[i]
                col_label = common_cols[j]
                old_val = old_aligned.iloc[i, j]
                new_val = new_aligned.iloc[i, j]
                result.modified_cells.append(
                    CellChange(
                        row=row_label,
                        column=col_label,
                        old_value=old_val,
                        new_value=new_val,
                        change_type=ChangeType.VALUE_MODIFIED,
                    )
                )

        # Track NA→value cells separately
        if na_to_value_count > 0 and na_to_value_count <= MAX_CELL_CHANGES:
            for i, j in zip(na_to_value_rows, na_to_value_cols, strict=True):
                row_label = common_idx[i]
                col_label = common_cols[j]
                old_val = old_aligned.iloc[i, j]
                new_val = new_aligned.iloc[i, j]
                result.na_to_value_cells.append(
                    CellChange(
                        row=row_label,
                        column=col_label,
                        old_value=old_val,
                        new_value=new_val,
                        change_type=ChangeType.VALUE_MODIFIED,
                    )
                )

        # Check NA type changes (only for cells where both are NA)
        if self.check_na_type:
            na_type_changes = self._check_na_type_changes_vectorized(
                old_aligned, new_aligned, both_na
            )
            result.na_type_changes_count = len(na_type_changes)
            if len(na_type_changes) <= MAX_CELL_CHANGES:
                result.na_type_changes = na_type_changes

        # Determine if identical
        result.is_identical = result.total_changes == 0

        return result

    def _check_na_type_changes_vectorized(
        self,
        old_aligned: pd.DataFrame,
        new_aligned: pd.DataFrame,
        both_na: pd.DataFrame,
    ) -> list[CellChange]:
        """Check for NA type differences in cells where both are NA.

        For non-object columns with matching dtypes, all NAs are the same type
        (e.g., float64 always uses np.nan, Float64 always uses pd.NA).
        NA type differences can only occur in object columns or when dtypes differ.
        Since dtype differences are already detected separately, we only check
        object columns here for performance.

        Args:
            old_aligned: Aligned old DataFrame
            new_aligned: Aligned new DataFrame
            both_na: Boolean mask of cells where both are NA

        Returns:
            List of CellChange for NA type changes
        """
        if not both_na.any().any():
            return []

        # Only object columns can have mixed NA types (None, pd.NA, np.nan)
        # Non-object columns always have consistent NA type based on dtype
        object_col_indices = set()
        for idx, col in enumerate(old_aligned.columns):
            if old_aligned[col].dtype == object or new_aligned[col].dtype == object:
                object_col_indices.add(idx)

        # Fast path: no object columns means no possible NA type differences
        if not object_col_indices:
            return []

        na_type_changes: list[CellChange] = []
        na_rows, na_cols = np.where(both_na.values)

        # Use .values for fast numpy indexing instead of .iloc
        old_vals = old_aligned.values
        new_vals = new_aligned.values

        for i, j in zip(na_rows, na_cols, strict=True):
            # Skip non-object columns (they have consistent NA types)
            if j not in object_col_indices:
                continue

            old_val = old_vals[i, j]
            new_val = new_vals[i, j]
            if _get_na_type(old_val) != _get_na_type(new_val):
                # Stop collecting after MAX_CELL_CHANGES (count will still be accurate)
                if len(na_type_changes) >= MAX_CELL_CHANGES:
                    continue
                row_label = old_aligned.index[i]
                col_label = old_aligned.columns[j]
                na_type_changes.append(
                    CellChange(
                        row=row_label,
                        column=col_label,
                        old_value=old_val,
                        new_value=new_val,
                        change_type=ChangeType.NA_TYPE_CHANGED,
                    )
                )

        return na_type_changes

    def _values_equal(
        self,
        old_val: Any,
        new_val: Any,
        dtype: Any,
    ) -> bool:
        """Compare two values with appropriate tolerance.

        Args:
            old_val: Old value
            new_val: New value
            dtype: Column dtype

        Returns:
            True if values are considered equal
        """
        # Handle NA cases
        old_is_na = pd.isna(old_val)
        new_is_na = pd.isna(new_val)

        if old_is_na and new_is_na:
            return True  # Both NA, considered equal (NA type checked separately)
        if old_is_na != new_is_na:
            return False  # One NA, one not

        # Numeric comparison with tolerance
        if pd.api.types.is_numeric_dtype(dtype):
            try:
                return bool(
                    np.isclose(
                        float(old_val),
                        float(new_val),
                        rtol=self.rtol,
                        atol=self.atol,
                    )
                )
            except (ValueError, TypeError):
                return old_val == new_val

        # Exact match for other types
        return old_val == new_val

    def _detect_na_type_change(
        self,
        old_val: Any,
        new_val: Any,
    ) -> bool:
        """Detect if NA type differs between values.

        Args:
            old_val: Old value
            new_val: New value

        Returns:
            True if both are NA but of different types
        """
        old_is_na = pd.isna(old_val)
        new_is_na = pd.isna(new_val)

        if not (old_is_na and new_is_na):
            return False

        old_type = _get_na_type(old_val)
        new_type = _get_na_type(new_val)

        return old_type != new_type


def _get_na_type(val: Any) -> str:
    """Identify the type of NA value.

    Args:
        val: Value to check

    Returns:
        String identifying NA type
    """
    if val is None:
        return "None"
    if val is pd.NA:
        return "pd.NA"
    if isinstance(val, float) and np.isnan(val):
        return "np.nan"
    return "not_na"
