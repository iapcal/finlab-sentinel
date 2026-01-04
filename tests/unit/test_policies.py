"""Tests for comparison policies."""

from finlab_sentinel.comparison.differ import ComparisonResult, DtypeChange
from finlab_sentinel.comparison.policies import (
    AppendOnlyPolicy,
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


class TestAppendOnlyPolicyWithNaToValue:
    """Tests for AppendOnlyPolicy with ignore_na_to_value."""

    def test_allows_na_to_value_when_ignored(self):
        """Verify NA→value changes are allowed when ignored."""
        from finlab_sentinel.comparison.differ import CellChange, ChangeType

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
