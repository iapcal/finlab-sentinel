"""DataFrame difference detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Set, Tuple

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


@dataclass
class ComparisonResult:
    """Result of DataFrame comparison."""

    is_identical: bool
    added_rows: Set[Any] = field(default_factory=set)
    deleted_rows: Set[Any] = field(default_factory=set)
    added_columns: Set[Any] = field(default_factory=set)
    deleted_columns: Set[Any] = field(default_factory=set)
    modified_cells: List[CellChange] = field(default_factory=list)
    dtype_changes: List[DtypeChange] = field(default_factory=list)
    na_type_changes: List[CellChange] = field(default_factory=list)

    # Metrics
    old_shape: Tuple[int, int] = (0, 0)
    new_shape: Tuple[int, int] = (0, 0)

    @property
    def total_changes(self) -> int:
        """Total number of changes detected."""
        return (
            len(self.added_rows)
            + len(self.deleted_rows)
            + len(self.added_columns)
            + len(self.deleted_columns)
            + len(self.modified_cells)
            + len(self.dtype_changes)
            + len(self.na_type_changes)
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
        # Modified cells
        modified = len(self.modified_cells) + len(self.na_type_changes)

        return (deleted_cells + modified) / total

    def is_append_only(self) -> bool:
        """Check if changes are append-only (no deletions/modifications)."""
        return (
            len(self.deleted_rows) == 0
            and len(self.deleted_columns) == 0
            and len(self.modified_cells) == 0
            and len(self.dtype_changes) == 0
            and len(self.na_type_changes) == 0
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
        if self.modified_cells:
            parts.append(f"{len(self.modified_cells)} modified cells")
        if self.dtype_changes:
            parts.append(f"{len(self.dtype_changes)} dtype changes")
        if self.na_type_changes:
            parts.append(f"{len(self.na_type_changes)} NA type changes")

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

        # Check for dtype changes in common columns
        common_columns = old_columns & new_columns
        if self.check_dtype:
            for col in common_columns:
                old_dtype = str(old_df[col].dtype)
                new_dtype = str(new_df[col].dtype)
                if old_dtype != new_dtype:
                    result.dtype_changes.append(
                        DtypeChange(col, old_dtype, new_dtype)
                    )

        # Compare values in common rows and columns
        common_index = old_index & new_index

        for row in common_index:
            for col in common_columns:
                old_val = old_df.loc[row, col]
                new_val = new_df.loc[row, col]

                # Check for NA type changes
                if self.check_na_type:
                    na_change = self._detect_na_type_change(old_val, new_val)
                    if na_change:
                        result.na_type_changes.append(
                            CellChange(
                                row=row,
                                column=col,
                                old_value=old_val,
                                new_value=new_val,
                                change_type=ChangeType.NA_TYPE_CHANGED,
                            )
                        )
                        continue

                # Check for value changes
                if not self._values_equal(old_val, new_val, old_df[col].dtype):
                    result.modified_cells.append(
                        CellChange(
                            row=row,
                            column=col,
                            old_value=old_val,
                            new_value=new_val,
                            change_type=ChangeType.VALUE_MODIFIED,
                        )
                    )

        # Determine if identical
        result.is_identical = result.total_changes == 0

        return result

    def _values_equal(
        self,
        old_val: Any,
        new_val: Any,
        dtype: np.dtype,
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
