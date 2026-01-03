"""Comparison policies for determining violations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Set

from finlab_sentinel.comparison.differ import ComparisonResult


class ComparisonPolicy(ABC):
    """Base class for comparison policies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name."""
        ...

    @abstractmethod
    def is_violation(self, result: ComparisonResult) -> bool:
        """Check if comparison result violates policy.

        Args:
            result: Comparison result to check

        Returns:
            True if policy is violated
        """
        ...

    @abstractmethod
    def get_violation_message(self, result: ComparisonResult) -> str:
        """Get human-readable violation message.

        Args:
            result: Comparison result

        Returns:
            Violation message
        """
        ...


class AppendOnlyPolicy(ComparisonPolicy):
    """Default policy: Only new dates/columns allowed.

    No deletion or modification of historical data.
    """

    @property
    def name(self) -> str:
        return "append_only"

    def is_violation(self, result: ComparisonResult) -> bool:
        """Check if any historical data was modified or deleted."""
        return not result.is_append_only()

    def get_violation_message(self, result: ComparisonResult) -> str:
        """Get detailed violation message."""
        violations = []

        if result.deleted_rows:
            violations.append(
                f"Deleted {len(result.deleted_rows)} rows: "
                f"{list(result.deleted_rows)[:5]}..."
                if len(result.deleted_rows) > 5
                else f"Deleted rows: {list(result.deleted_rows)}"
            )

        if result.deleted_columns:
            violations.append(
                f"Deleted {len(result.deleted_columns)} columns: "
                f"{list(result.deleted_columns)[:5]}..."
                if len(result.deleted_columns) > 5
                else f"Deleted columns: {list(result.deleted_columns)}"
            )

        if result.modified_cells:
            sample = result.modified_cells[:3]
            violations.append(
                f"{len(result.modified_cells)} cells modified. "
                f"Examples: {[str(c) for c in sample]}"
            )

        if result.dtype_changes:
            changes = [str(c) for c in result.dtype_changes[:3]]
            violations.append(f"Dtype changes: {changes}")

        if result.na_type_changes:
            violations.append(
                f"{len(result.na_type_changes)} NA type changes detected"
            )

        return (
            f"Append-only policy violation: {'; '.join(violations)}"
            if violations
            else "No violations"
        )


class ThresholdPolicy(ComparisonPolicy):
    """Policy that allows changes up to a threshold."""

    def __init__(self, threshold: float = 0.10) -> None:
        """Initialize threshold policy.

        Args:
            threshold: Maximum allowed change ratio (0.0 to 1.0)
        """
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "threshold"

    def is_violation(self, result: ComparisonResult) -> bool:
        """Check if change ratio exceeds threshold."""
        return result.exceeds_threshold(self.threshold)

    def get_violation_message(self, result: ComparisonResult) -> str:
        """Get threshold violation message."""
        return (
            f"Change ratio {result.change_ratio:.1%} exceeds "
            f"threshold {self.threshold:.1%}. {result.summary()}"
        )


class PermissivePolicy(ComparisonPolicy):
    """Permissive policy that allows all changes.

    Used for datasets in the history_modifiable blacklist.
    """

    @property
    def name(self) -> str:
        return "permissive"

    def is_violation(self, result: ComparisonResult) -> bool:
        """Never violates - all changes allowed."""
        return False

    def get_violation_message(self, result: ComparisonResult) -> str:
        """No violation possible."""
        return "Permissive policy: all changes allowed"


class CompositePolicy(ComparisonPolicy):
    """Combines multiple policies with AND logic.

    Violation occurs if ANY policy is violated.
    """

    def __init__(self, policies: list[ComparisonPolicy]) -> None:
        """Initialize composite policy.

        Args:
            policies: List of policies to combine
        """
        self.policies = policies

    @property
    def name(self) -> str:
        return f"composite({', '.join(p.name for p in self.policies)})"

    def is_violation(self, result: ComparisonResult) -> bool:
        """Check if any policy is violated."""
        return any(p.is_violation(result) for p in self.policies)

    def get_violation_message(self, result: ComparisonResult) -> str:
        """Get combined violation messages."""
        violations = [
            p.get_violation_message(result)
            for p in self.policies
            if p.is_violation(result)
        ]
        return " | ".join(violations) if violations else "No violations"


def get_policy_for_dataset(
    dataset: str,
    default_mode: str,
    history_modifiable: Set[str],
    threshold: float = 0.10,
) -> ComparisonPolicy:
    """Get appropriate policy for a dataset.

    Args:
        dataset: Dataset name
        default_mode: Default policy mode
        history_modifiable: Set of datasets that can modify history
        threshold: Threshold for threshold policy

    Returns:
        Appropriate policy for the dataset
    """
    # Check if dataset is in blacklist (history modifiable)
    if dataset in history_modifiable:
        return PermissivePolicy()

    # Return policy based on default mode
    if default_mode == "append_only":
        return AppendOnlyPolicy()
    elif default_mode == "threshold":
        return ThresholdPolicy(threshold)
    elif default_mode == "permissive":
        return PermissivePolicy()
    else:
        # Default to append_only
        return AppendOnlyPolicy()
