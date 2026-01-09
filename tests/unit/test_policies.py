"""Tests for comparison policies."""

from finlab_sentinel.comparison.differ import (
    CellChange,
    ChangeType,
    ComparisonResult,
    DtypeChange,
)
from finlab_sentinel.comparison.policies import (
    AppendOnlyPolicy,
    CompositePolicy,
    PermissivePolicy,
    ThresholdPolicy,
    get_policy_for_dataset,
)


class TestAppendOnlyPolicy:
    """Tests for AppendOnlyPolicy."""

    def test_allows_new_rows(self):
        """Verify new rows don't violate policy."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            added_rows={"2025-01-11", "2025-01-12"},
            old_shape=(10, 4),
            new_shape=(12, 4),
        )

        assert not policy.is_violation(result)

    def test_allows_new_columns(self):
        """Verify new columns don't violate policy."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            added_columns={"NEW_COL"},
            old_shape=(10, 4),
            new_shape=(10, 5),
        )

        assert not policy.is_violation(result)

    def test_rejects_deleted_rows(self):
        """Verify deleted rows violate policy."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            deleted_rows={"2025-01-01"},
            old_shape=(10, 4),
            new_shape=(9, 4),
        )

        assert policy.is_violation(result)

    def test_rejects_deleted_columns(self):
        """Verify deleted columns violate policy."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            deleted_columns={"2330"},
            old_shape=(10, 4),
            new_shape=(10, 3),
        )

        assert policy.is_violation(result)

    def test_rejects_modified_values(self):
        """Verify modified values violate policy."""
        from finlab_sentinel.comparison.differ import CellChange, ChangeType

        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            modified_cells=[
                CellChange(
                    row="2025-01-01",
                    column="2330",
                    old_value=100.0,
                    new_value=200.0,
                    change_type=ChangeType.VALUE_MODIFIED,
                )
            ],
            old_shape=(10, 4),
            new_shape=(10, 4),
        )

        assert policy.is_violation(result)

    def test_rejects_dtype_changes(self):
        """Verify dtype changes violate policy."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            dtype_changes=[DtypeChange("2330", "float64", "float32")],
            old_shape=(10, 4),
            new_shape=(10, 4),
        )

        assert policy.is_violation(result)


class TestThresholdPolicy:
    """Tests for ThresholdPolicy."""

    def test_allows_below_threshold(self):
        """Verify changes below threshold are allowed."""
        policy = ThresholdPolicy(threshold=0.10)

        result = ComparisonResult(
            is_identical=False,
            old_shape=(100, 10),
            new_shape=(100, 10),
        )
        # No changes, ratio = 0
        assert not policy.is_violation(result)

    def test_rejects_above_threshold(self):
        """Verify changes above threshold are rejected."""
        policy = ThresholdPolicy(threshold=0.10)

        result = ComparisonResult(
            is_identical=False,
            deleted_rows=set(range(20)),  # 20 deleted rows
            old_shape=(100, 10),  # 1000 cells total
            new_shape=(80, 10),
        )
        # 20 rows × 10 columns = 200 cells = 20% > 10% threshold

        assert policy.is_violation(result)


class TestPermissivePolicy:
    """Tests for PermissivePolicy."""

    def test_allows_all_changes(self):
        """Verify all changes are allowed."""
        policy = PermissivePolicy()

        result = ComparisonResult(
            is_identical=False,
            deleted_rows={"row1"},
            deleted_columns={"col1"},
            old_shape=(10, 4),
            new_shape=(9, 3),
        )

        assert not policy.is_violation(result)


class TestGetPolicyForDataset:
    """Tests for get_policy_for_dataset function."""

    def test_blacklisted_dataset_gets_permissive(self):
        """Verify blacklisted datasets get permissive policy."""
        policy = get_policy_for_dataset(
            dataset="fundamental:eps",
            default_mode="append_only",
            history_modifiable={"fundamental:eps"},
        )

        assert isinstance(policy, PermissivePolicy)

    def test_default_mode_append_only(self):
        """Verify default append_only mode."""
        policy = get_policy_for_dataset(
            dataset="price:close",
            default_mode="append_only",
            history_modifiable=set(),
        )

        assert isinstance(policy, AppendOnlyPolicy)

    def test_default_mode_threshold(self):
        """Verify threshold mode."""
        policy = get_policy_for_dataset(
            dataset="price:close",
            default_mode="threshold",
            history_modifiable=set(),
            threshold=0.15,
        )

        assert isinstance(policy, ThresholdPolicy)
        assert policy.threshold == 0.15

    def test_allow_na_to_value_whitelist(self):
        """Verify datasets in allow_na_to_value get ignore_na_to_value=True."""
        policy = get_policy_for_dataset(
            dataset="price:close",
            default_mode="append_only",
            history_modifiable=set(),
            allow_na_to_value={"price:close"},
        )

        assert isinstance(policy, AppendOnlyPolicy)
        assert policy.ignore_na_to_value is True

    def test_non_whitelisted_dataset_no_ignore(self):
        """Verify non-whitelisted datasets don't ignore NA→value."""
        policy = get_policy_for_dataset(
            dataset="price:open",
            default_mode="append_only",
            history_modifiable=set(),
            allow_na_to_value={"price:close"},
        )

        assert isinstance(policy, AppendOnlyPolicy)
        assert policy.ignore_na_to_value is False

    def test_default_mode_permissive(self):
        """Verify permissive mode returns PermissivePolicy."""
        policy = get_policy_for_dataset(
            dataset="price:close",
            default_mode="permissive",
            history_modifiable=set(),
        )

        assert isinstance(policy, PermissivePolicy)

    def test_unknown_mode_defaults_to_append_only(self):
        """Verify unknown mode defaults to append_only."""
        policy = get_policy_for_dataset(
            dataset="price:close",
            default_mode="unknown_mode",
            history_modifiable=set(),
        )

        assert isinstance(policy, AppendOnlyPolicy)


class TestAppendOnlyPolicyViolationMessages:
    """Tests for AppendOnlyPolicy violation message formatting."""

    def test_violation_message_many_deleted_rows(self):
        """Verify violation message truncates long row lists."""
        policy = AppendOnlyPolicy()

        # Create result with more than 5 deleted rows
        result = ComparisonResult(
            is_identical=False,
            deleted_rows=set(range(10)),  # 10 deleted rows
            old_shape=(20, 4),
            new_shape=(10, 4),
        )

        message = policy.get_violation_message(result)
        assert "Deleted 10 rows" in message
        assert "..." in message

    def test_violation_message_few_deleted_rows(self):
        """Verify violation message shows all rows when few."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            deleted_rows={"row1", "row2"},
            old_shape=(10, 4),
            new_shape=(8, 4),
        )

        message = policy.get_violation_message(result)
        assert "Deleted rows:" in message
        assert "..." not in message

    def test_violation_message_many_deleted_columns(self):
        """Verify violation message truncates long column lists."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            deleted_columns={f"col{i}" for i in range(10)},
            old_shape=(10, 15),
            new_shape=(10, 5),
        )

        message = policy.get_violation_message(result)
        assert "Deleted 10 columns" in message
        assert "..." in message

    def test_violation_message_few_deleted_columns(self):
        """Verify violation message shows all columns when few."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            deleted_columns={"col1", "col2"},
            old_shape=(10, 6),
            new_shape=(10, 4),
        )

        message = policy.get_violation_message(result)
        assert "Deleted columns:" in message

    def test_violation_message_modified_cells(self):
        """Verify violation message includes modified cells info."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            modified_cells=[
                CellChange("row1", "col1", 1.0, 2.0, ChangeType.VALUE_MODIFIED),
            ],
            old_shape=(10, 4),
            new_shape=(10, 4),
        )

        message = policy.get_violation_message(result)
        assert "cells modified" in message

    def test_violation_message_dtype_changes(self):
        """Verify violation message includes dtype changes."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            dtype_changes=[DtypeChange("col1", "float64", "int64")],
            old_shape=(10, 4),
            new_shape=(10, 4),
        )

        message = policy.get_violation_message(result)
        assert "Dtype changes" in message

    def test_violation_message_na_type_changes(self):
        """Verify violation message includes NA type changes."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=False,
            na_type_changes=[
                CellChange(
                    "row1", "col1", None, float("nan"), ChangeType.NA_TYPE_CHANGED
                )
            ],
            na_type_changes_count=1,
            old_shape=(10, 4),
            new_shape=(10, 4),
        )

        message = policy.get_violation_message(result)
        assert "NA type changes" in message

    def test_violation_message_no_violations(self):
        """Verify message when no violations."""
        policy = AppendOnlyPolicy()

        result = ComparisonResult(
            is_identical=True,
            old_shape=(10, 4),
            new_shape=(10, 4),
        )

        message = policy.get_violation_message(result)
        assert message == "No violations"


class TestThresholdPolicyExtended:
    """Extended tests for ThresholdPolicy."""

    def test_name_property(self):
        """Verify name property returns correct value."""
        policy = ThresholdPolicy(threshold=0.10)
        assert policy.name == "threshold"

    def test_get_violation_message(self):
        """Verify violation message includes ratio and threshold."""
        policy = ThresholdPolicy(threshold=0.10)

        result = ComparisonResult(
            is_identical=False,
            deleted_rows=set(range(20)),
            old_shape=(100, 10),
            new_shape=(80, 10),
        )

        message = policy.get_violation_message(result)
        assert "exceeds threshold" in message
        assert "10.0%" in message  # threshold


class TestPermissivePolicyExtended:
    """Extended tests for PermissivePolicy."""

    def test_name_property(self):
        """Verify name property returns correct value."""
        policy = PermissivePolicy()
        assert policy.name == "permissive"

    def test_get_violation_message(self):
        """Verify violation message for permissive policy."""
        policy = PermissivePolicy()

        result = ComparisonResult(
            is_identical=False,
            deleted_rows={"row1"},
            old_shape=(10, 4),
            new_shape=(9, 4),
        )

        message = policy.get_violation_message(result)
        assert "all changes allowed" in message


class TestCompositePolicy:
    """Tests for CompositePolicy."""

    def test_name_combines_policy_names(self):
        """Verify name includes all policy names."""
        policy = CompositePolicy([AppendOnlyPolicy(), ThresholdPolicy(0.10)])

        assert "append_only" in policy.name
        assert "threshold" in policy.name
        assert "composite" in policy.name

    def test_violation_when_any_policy_violated(self):
        """Verify violation occurs if any policy is violated."""
        # Threshold policy with low threshold
        policy = CompositePolicy([AppendOnlyPolicy(), ThresholdPolicy(0.05)])

        # This violates append_only (deleted rows)
        result = ComparisonResult(
            is_identical=False,
            deleted_rows={"row1"},
            old_shape=(10, 4),
            new_shape=(9, 4),
        )

        assert policy.is_violation(result) is True

    def test_no_violation_when_all_policies_pass(self):
        """Verify no violation when all policies pass."""
        policy = CompositePolicy(
            [
                PermissivePolicy(),  # Always passes
            ]
        )

        result = ComparisonResult(
            is_identical=False,
            deleted_rows={"row1"},
            old_shape=(10, 4),
            new_shape=(9, 4),
        )

        assert policy.is_violation(result) is False

    def test_get_violation_message_combines_messages(self):
        """Verify violation message combines all violated policies."""
        policy = CompositePolicy(
            [
                AppendOnlyPolicy(),
                ThresholdPolicy(0.01),  # Very low threshold
            ]
        )

        # This violates both policies
        result = ComparisonResult(
            is_identical=False,
            deleted_rows=set(range(50)),  # 50 deleted rows
            old_shape=(100, 10),
            new_shape=(50, 10),
        )

        message = policy.get_violation_message(result)
        assert "Append-only policy violation" in message
        assert "exceeds threshold" in message
        assert "|" in message  # separator

    def test_get_violation_message_no_violations(self):
        """Verify message when no violations."""
        policy = CompositePolicy([PermissivePolicy()])

        result = ComparisonResult(
            is_identical=False,
            deleted_rows={"row1"},
            old_shape=(10, 4),
            new_shape=(9, 4),
        )

        message = policy.get_violation_message(result)
        assert message == "No violations"


class TestAppendOnlyPolicyWithNaToValue:
    """Tests for AppendOnlyPolicy with ignore_na_to_value."""

    def test_allows_na_to_value_when_ignored(self):
        """Verify NA→value changes are allowed when ignored."""
        policy = AppendOnlyPolicy(ignore_na_to_value=True)

        # Result with only NA→value modifications
        result = ComparisonResult(
            is_identical=False,
            modified_cells=[
                CellChange(
                    row="2025-01-01",
                    column="col",
                    old_value=None,
                    new_value=100.0,
                    change_type=ChangeType.VALUE_MODIFIED,
                )
            ],
            modified_cells_count=1,
            na_to_value_cells_count=1,
            old_shape=(10, 4),
            new_shape=(10, 4),
        )

        assert not policy.is_violation(result)

    def test_rejects_non_na_modifications_even_with_ignore(self):
        """Verify normal modifications still violate even with ignore_na_to_value."""
        from finlab_sentinel.comparison.differ import CellChange, ChangeType

        policy = AppendOnlyPolicy(ignore_na_to_value=True)

        # Result with regular value modifications (not NA→value)
        result = ComparisonResult(
            is_identical=False,
            modified_cells=[
                CellChange(
                    row="2025-01-01",
                    column="col",
                    old_value=50.0,
                    new_value=100.0,
                    change_type=ChangeType.VALUE_MODIFIED,
                )
            ],
            modified_cells_count=1,
            na_to_value_cells_count=0,  # Not a NA→value change
            old_shape=(10, 4),
            new_shape=(10, 4),
        )

        assert policy.is_violation(result)
