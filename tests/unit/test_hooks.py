"""Tests for preprocess hooks."""

import pandas as pd
import pytest

from finlab_sentinel.core.hooks import (
    PreprocessHookRegistry,
    clear_preprocess_hooks,
    get_registry,
    register_preprocess_hook,
    unregister_preprocess_hook,
)


@pytest.fixture
def registry() -> PreprocessHookRegistry:
    """Create a fresh registry for each test."""
    return PreprocessHookRegistry()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "A": [1.123456, 2.234567, 3.345678],
            "B": [4.456789, 5.567890, 6.678901],
        }
    )


class TestPreprocessHookRegistry:
    """Tests for PreprocessHookRegistry class."""

    def test_register_exact_hook(self, registry: PreprocessHookRegistry):
        """Verify exact match hook registration."""
        hook = lambda df: df.round(2)
        registry.register("price:收盤價", hook)

        assert registry.get("price:收盤價") is hook
        assert registry.get("price:開盤價") is None

    def test_register_pattern_hook(self, registry: PreprocessHookRegistry):
        """Verify pattern hook registration with wildcards."""
        hook = lambda df: df.round(2)
        registry.register("price:*", hook)

        assert registry.get("price:收盤價") is hook
        assert registry.get("price:開盤價") is hook
        assert registry.get("fundamental:營收") is None

    def test_exact_takes_precedence(self, registry: PreprocessHookRegistry):
        """Verify exact match takes precedence over pattern."""
        pattern_hook = lambda df: df.round(1)
        exact_hook = lambda df: df.round(2)

        registry.register("price:*", pattern_hook)
        registry.register("price:收盤價", exact_hook)

        assert registry.get("price:收盤價") is exact_hook
        assert registry.get("price:開盤價") is pattern_hook

    def test_unregister_exact(self, registry: PreprocessHookRegistry):
        """Verify unregistering exact hook."""
        hook = lambda df: df.round(2)
        registry.register("price:收盤價", hook)

        result = registry.unregister("price:收盤價")

        assert result is True
        assert registry.get("price:收盤價") is None

    def test_unregister_pattern(self, registry: PreprocessHookRegistry):
        """Verify unregistering pattern hook."""
        hook = lambda df: df.round(2)
        registry.register("price:*", hook)

        result = registry.unregister("price:*")

        assert result is True
        assert registry.get("price:收盤價") is None

    def test_unregister_nonexistent(self, registry: PreprocessHookRegistry):
        """Verify unregistering non-existent hook returns False."""
        result = registry.unregister("nonexistent")
        assert result is False

    def test_apply_with_hook(
        self, registry: PreprocessHookRegistry, sample_df: pd.DataFrame
    ):
        """Verify apply returns preprocessed DataFrame when hook exists."""
        registry.register("test", lambda df: df.round(2))

        result = registry.apply("test", sample_df)

        expected = sample_df.round(2)
        pd.testing.assert_frame_equal(result, expected)

    def test_apply_without_hook(
        self, registry: PreprocessHookRegistry, sample_df: pd.DataFrame
    ):
        """Verify apply returns original DataFrame when no hook."""
        result = registry.apply("test", sample_df)

        pd.testing.assert_frame_equal(result, sample_df)

    def test_apply_handles_error(
        self, registry: PreprocessHookRegistry, sample_df: pd.DataFrame
    ):
        """Verify apply returns original DataFrame when hook raises error."""

        def bad_hook(df: pd.DataFrame) -> pd.DataFrame:
            raise ValueError("intentional error")

        registry.register("test", bad_hook)

        result = registry.apply("test", sample_df)

        # Should return original data on error
        pd.testing.assert_frame_equal(result, sample_df)

    def test_apply_handles_wrong_return_type(
        self, registry: PreprocessHookRegistry, sample_df: pd.DataFrame
    ):
        """Verify apply returns original DataFrame when hook returns non-DataFrame."""
        registry.register("test", lambda df: "not a dataframe")

        result = registry.apply("test", sample_df)

        # Should return original data when hook returns wrong type
        pd.testing.assert_frame_equal(result, sample_df)

    def test_clear(self, registry: PreprocessHookRegistry):
        """Verify clear removes all hooks."""
        registry.register("exact", lambda df: df)
        registry.register("pattern:*", lambda df: df)

        registry.clear()

        assert registry.get("exact") is None
        assert registry.get("pattern:test") is None

    def test_list_hooks(self, registry: PreprocessHookRegistry):
        """Verify list_hooks returns all registered hooks."""
        registry.register("exact1", lambda df: df)
        registry.register("exact2", lambda df: df)
        registry.register("pattern:*", lambda df: df)

        hooks = registry.list_hooks()

        assert set(hooks["exact"]) == {"exact1", "exact2"}
        assert hooks["pattern"] == ["pattern:*"]

    def test_question_mark_wildcard(self, registry: PreprocessHookRegistry):
        """Verify ? wildcard matches single character."""
        hook = lambda df: df.round(2)
        registry.register("price:?", hook)

        assert registry.get("price:A") is hook
        assert registry.get("price:AB") is None


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def setup_method(self):
        """Clear global registry before each test."""
        clear_preprocess_hooks()

    def teardown_method(self):
        """Clear global registry after each test."""
        clear_preprocess_hooks()

    def test_register_and_get(self, sample_df: pd.DataFrame):
        """Verify register and get from global registry."""
        hook = lambda df: df.round(2)
        register_preprocess_hook("test", hook)

        registry = get_registry()
        result = registry.apply("test", sample_df)

        expected = sample_df.round(2)
        pd.testing.assert_frame_equal(result, expected)

    def test_unregister(self):
        """Verify unregister from global registry."""
        register_preprocess_hook("test", lambda df: df)

        result = unregister_preprocess_hook("test")

        assert result is True
        assert get_registry().get("test") is None

    def test_clear(self):
        """Verify clear global registry."""
        register_preprocess_hook("test", lambda df: df)

        clear_preprocess_hooks()

        assert get_registry().get("test") is None


class TestPreprocessHookIntegration:
    """Integration tests for preprocess hooks with interceptor."""

    def setup_method(self):
        """Clear hooks before each test."""
        clear_preprocess_hooks()

    def teardown_method(self):
        """Clear hooks after each test."""
        clear_preprocess_hooks()

    def test_hook_applied_to_dataframe(self, sample_df: pd.DataFrame):
        """Verify hook is applied correctly."""
        # Register a hook that rounds to 2 decimal places
        register_preprocess_hook("price:*", lambda df: df.round(2))

        registry = get_registry()
        result = registry.apply("price:收盤價", sample_df)

        # Verify rounding was applied
        assert result["A"].iloc[0] == 1.12
        assert result["B"].iloc[0] == 4.46

    def test_hook_sorting_columns(self, sample_df: pd.DataFrame):
        """Verify hook can sort columns for consistent comparison."""
        df_unsorted = sample_df[["B", "A"]]

        register_preprocess_hook("test", lambda df: df[sorted(df.columns)])

        registry = get_registry()
        result = registry.apply("test", df_unsorted)

        assert list(result.columns) == ["A", "B"]
