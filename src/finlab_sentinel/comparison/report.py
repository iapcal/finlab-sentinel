"""Anomaly report generation and storage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from finlab_sentinel.comparison.differ import (
    CellChange,
    ChangeType,
    ComparisonResult,
    DtypeChange,
)

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Detailed report of a data anomaly."""

    dataset: str
    backup_key: str
    detected_at: datetime
    comparison_result: ComparisonResult
    policy_name: str
    violation_message: str

    # Optional metadata
    old_hash: str | None = None
    new_hash: str | None = None

    @property
    def summary(self) -> str:
        """Get short summary of the anomaly."""
        return (
            f"{self.dataset}: {self.comparison_result.summary()} "
            f"[{self.policy_name} violated]"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        result = self.comparison_result

        def serialize_change(change: CellChange) -> dict:
            return {
                "row": str(change.row),
                "column": str(change.column),
                "old_value": _serialize_value(change.old_value),
                "new_value": _serialize_value(change.new_value),
                "change_type": change.change_type.value,
            }

        def serialize_dtype_change(change: DtypeChange) -> dict:
            return {
                "column": str(change.column),
                "old_dtype": change.old_dtype,
                "new_dtype": change.new_dtype,
            }

        return {
            "dataset": self.dataset,
            "backup_key": self.backup_key,
            "detected_at": self.detected_at.isoformat(),
            "policy_name": self.policy_name,
            "violation_message": self.violation_message,
            "old_hash": self.old_hash,
            "new_hash": self.new_hash,
            "comparison": {
                "is_identical": result.is_identical,
                "old_shape": list(result.old_shape),
                "new_shape": list(result.new_shape),
                "change_ratio": result.change_ratio,
                "added_rows": [str(r) for r in result.added_rows],
                "deleted_rows": [str(r) for r in result.deleted_rows],
                "added_columns": [str(c) for c in result.added_columns],
                "deleted_columns": [str(c) for c in result.deleted_columns],
                "modified_cells": [
                    serialize_change(c) for c in result.modified_cells[:100]
                ],  # Limit to 100
                "dtype_changes": [
                    serialize_dtype_change(c) for c in result.dtype_changes
                ],
                "na_type_changes": [
                    serialize_change(c) for c in result.na_type_changes[:100]
                ],
                "total_changes": result.total_changes,
            },
            "summary": self.summary,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, directory: Path) -> Path:
        """Save report to file.

        Args:
            directory: Directory to save report

        Returns:
            Path to saved report file
        """
        directory.mkdir(parents=True, exist_ok=True)

        # Generate filename
        timestamp = self.detected_at.strftime("%Y-%m-%d_%H%M%S")
        safe_key = self.backup_key.replace("/", "_").replace("\\", "_")
        filename = f"{timestamp}_{safe_key}_anomaly.json"

        filepath = directory / filename

        filepath.write_text(self.to_json(), encoding="utf-8")

        logger.info(f"Saved anomaly report to: {filepath}")

        return filepath

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnomalyReport:
        """Create report from dictionary.

        Args:
            data: Dictionary with report data

        Returns:
            AnomalyReport instance
        """
        comparison_data = data["comparison"]

        # Reconstruct CellChanges
        modified_cells = [
            CellChange(
                row=c["row"],
                column=c["column"],
                old_value=c["old_value"],
                new_value=c["new_value"],
                change_type=ChangeType(c["change_type"]),
            )
            for c in comparison_data.get("modified_cells", [])
        ]

        na_type_changes = [
            CellChange(
                row=c["row"],
                column=c["column"],
                old_value=c["old_value"],
                new_value=c["new_value"],
                change_type=ChangeType(c["change_type"]),
            )
            for c in comparison_data.get("na_type_changes", [])
        ]

        dtype_changes = [
            DtypeChange(
                column=c["column"],
                old_dtype=c["old_dtype"],
                new_dtype=c["new_dtype"],
            )
            for c in comparison_data.get("dtype_changes", [])
        ]

        result = ComparisonResult(
            is_identical=comparison_data["is_identical"],
            old_shape=tuple(comparison_data["old_shape"]),
            new_shape=tuple(comparison_data["new_shape"]),
            added_rows=set(comparison_data.get("added_rows", [])),
            deleted_rows=set(comparison_data.get("deleted_rows", [])),
            added_columns=set(comparison_data.get("added_columns", [])),
            deleted_columns=set(comparison_data.get("deleted_columns", [])),
            modified_cells=modified_cells,
            dtype_changes=dtype_changes,
            na_type_changes=na_type_changes,
        )

        return cls(
            dataset=data["dataset"],
            backup_key=data["backup_key"],
            detected_at=datetime.fromisoformat(data["detected_at"]),
            comparison_result=result,
            policy_name=data["policy_name"],
            violation_message=data["violation_message"],
            old_hash=data.get("old_hash"),
            new_hash=data.get("new_hash"),
        )

    @classmethod
    def load(cls, filepath: Path) -> AnomalyReport:
        """Load report from file.

        Args:
            filepath: Path to report file

        Returns:
            AnomalyReport instance
        """
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def list_reports(directory: Path) -> list[Path]:
    """List all anomaly report files in directory.

    Args:
        directory: Directory to search

    Returns:
        List of report file paths, sorted by date (newest first)
    """
    if not directory.exists():
        return []

    reports = list(directory.glob("*_anomaly.json"))
    reports.sort(reverse=True)

    return reports


def _serialize_value(val: Any) -> Any:
    """Serialize a value for JSON.

    Args:
        val: Value to serialize

    Returns:
        JSON-serializable value
    """
    import numpy as np
    import pandas as pd

    # Check for numpy array first (before pd.isna which doesn't work on arrays)
    if isinstance(val, np.ndarray):
        return val.tolist()

    if isinstance(val, (np.integer, np.floating)):
        return float(val)

    if isinstance(val, (datetime, pd.Timestamp)):
        return val.isoformat()

    # Check for NA values (scalar only)
    try:
        if pd.isna(val):
            if val is None:
                return {"__na__": "None"}
            if val is pd.NA:
                return {"__na__": "pd.NA"}
            if isinstance(val, float):
                return {"__na__": "np.nan"}
            return {"__na__": "unknown"}
    except (ValueError, TypeError):
        pass  # Not a scalar NA value

    return str(val)
