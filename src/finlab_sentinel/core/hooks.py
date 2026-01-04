"""Preprocess hooks for data comparison."""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Callable
from typing import TypeAlias

import pandas as pd

logger = logging.getLogger(__name__)

# Type alias for preprocess function
PreprocessFn: TypeAlias = Callable[[pd.DataFrame], pd.DataFrame]


class PreprocessHookRegistry:
    """Registry for dataset-specific preprocess hooks.

    Preprocess hooks are applied before comparison to normalize data.
    This is useful for ignoring expected variations (e.g., rounding,
    column ordering) that shouldn't trigger anomaly detection.

    The original data is always returned to the user; preprocess is
    only used for comparison and backup storage.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        # Exact match hooks: dataset -> function
        self._exact_hooks: dict[str, PreprocessFn] = {}
        # Pattern hooks: pattern -> function (supports wildcards)
        self._pattern_hooks: dict[str, PreprocessFn] = {}

    def register(
        self,
        dataset: str,
        fn: PreprocessFn,
    ) -> None:
        """Register a preprocess hook for a dataset.

        Args:
            dataset: Dataset name or pattern (supports * and ? wildcards)
            fn: Function that takes DataFrame and returns preprocessed DataFrame

        Examples:
            >>> registry.register("price:收盤價", lambda df: df.round(2))
            >>> registry.register("price:*", my_price_preprocessor)
            >>> registry.register("fundamental_features:*", my_fundamental_preprocessor)
        """
        if "*" in dataset or "?" in dataset:
            self._pattern_hooks[dataset] = fn
            logger.debug(f"Registered pattern preprocess hook: {dataset}")
        else:
            self._exact_hooks[dataset] = fn
            logger.debug(f"Registered exact preprocess hook: {dataset}")

    def unregister(self, dataset: str) -> bool:
        """Unregister a preprocess hook.

        Args:
            dataset: Dataset name or pattern to unregister

        Returns:
            True if hook was found and removed, False otherwise
        """
        if dataset in self._exact_hooks:
            del self._exact_hooks[dataset]
            logger.debug(f"Unregistered exact preprocess hook: {dataset}")
            return True
        if dataset in self._pattern_hooks:
            del self._pattern_hooks[dataset]
            logger.debug(f"Unregistered pattern preprocess hook: {dataset}")
            return True
        return False

    def get(self, dataset: str) -> PreprocessFn | None:
        """Get preprocess hook for a dataset.

        Exact match takes precedence over pattern match.
        If multiple patterns match, the first registered one is used.

        Args:
            dataset: Dataset name

        Returns:
            Preprocess function or None if no hook registered
        """
        # Exact match first
        if dataset in self._exact_hooks:
            return self._exact_hooks[dataset]

        # Pattern match
        for pattern, fn in self._pattern_hooks.items():
            if fnmatch.fnmatch(dataset, pattern):
                return fn

        return None

    def apply(self, dataset: str, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocess hook to DataFrame if one exists.

        Args:
            dataset: Dataset name
            df: DataFrame to preprocess

        Returns:
            Preprocessed DataFrame, or original if no hook registered
        """
        fn = self.get(dataset)
        if fn is None:
            return df

        try:
            result = fn(df)
            if not isinstance(result, pd.DataFrame):
                logger.warning(
                    f"Preprocess hook for {dataset} did not return DataFrame, "
                    f"got {type(result).__name__}. Using original data."
                )
                return df
            logger.debug(f"Applied preprocess hook for {dataset}")
            return result
        except Exception as e:
            logger.error(
                f"Preprocess hook failed for {dataset}: {e}. Using original data."
            )
            return df

    def clear(self) -> None:
        """Clear all registered hooks."""
        self._exact_hooks.clear()
        self._pattern_hooks.clear()
        logger.debug("Cleared all preprocess hooks")

    def list_hooks(self) -> dict[str, list[str]]:
        """List all registered hooks.

        Returns:
            Dict with 'exact' and 'pattern' keys containing dataset names
        """
        return {
            "exact": list(self._exact_hooks.keys()),
            "pattern": list(self._pattern_hooks.keys()),
        }


# Global registry instance
_registry = PreprocessHookRegistry()


def get_registry() -> PreprocessHookRegistry:
    """Get the global preprocess hook registry.

    Returns:
        Global PreprocessHookRegistry instance
    """
    return _registry


def register_preprocess_hook(dataset: str, fn: PreprocessFn) -> None:
    """Register a preprocess hook for a dataset.

    Convenience function that uses the global registry.

    Args:
        dataset: Dataset name or pattern (supports * and ? wildcards)
        fn: Function that takes DataFrame and returns preprocessed DataFrame

    Example:
        >>> import finlab_sentinel
        >>> finlab_sentinel.register_preprocess_hook(
        ...     "price:收盤價",
        ...     lambda df: df.round(2)
        ... )
    """
    _registry.register(dataset, fn)


def unregister_preprocess_hook(dataset: str) -> bool:
    """Unregister a preprocess hook.

    Args:
        dataset: Dataset name or pattern to unregister

    Returns:
        True if hook was found and removed, False otherwise
    """
    return _registry.unregister(dataset)


def clear_preprocess_hooks() -> None:
    """Clear all registered preprocess hooks."""
    _registry.clear()
