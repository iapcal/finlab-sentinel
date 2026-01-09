"""Data interception and orchestration."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pandas as pd

from finlab_sentinel.comparison.differ import DataFrameComparer
from finlab_sentinel.comparison.hasher import ContentHasher
from finlab_sentinel.comparison.policies import get_policy_for_dataset
from finlab_sentinel.comparison.report import AnomalyReport
from finlab_sentinel.config.schema import SentinelConfig
from finlab_sentinel.core.hooks import get_registry as get_preprocess_registry
from finlab_sentinel.core.time_travel import TimeTravelContext
from finlab_sentinel.exceptions import NoHistoricalDataError
from finlab_sentinel.handlers.callback import create_handler_from_config
from finlab_sentinel.storage.parquet import ParquetStorage, sanitize_backup_key

logger = logging.getLogger(__name__)


class DataInterceptor:
    """Intercepts data.get calls and performs comparison logic."""

    def __init__(
        self,
        original_get: Callable,
        config: SentinelConfig,
    ) -> None:
        """Initialize interceptor.

        Args:
            original_get: Original data.get function
            config: Sentinel configuration
        """
        self.original_get = original_get
        self.config = config

        # Initialize components
        self.storage = ParquetStorage(
            base_path=config.get_storage_path(),
            compression=config.storage.compression,
        )

        self.hasher = ContentHasher()

        self.comparer = DataFrameComparer(
            rtol=config.comparison.rtol,
            atol=config.comparison.atol,
            check_dtype=config.comparison.check_dtype,
            check_na_type=config.comparison.check_na_type,
        )

        # Create handler
        self.handler = create_handler_from_config(
            behavior=config.anomaly.behavior.value,
            callback=config.anomaly.get_callback(),
        )

    def __call__(
        self,
        dataset: str,
        *args: Any,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Intercept data.get call.

        Args:
            dataset: Dataset name
            *args: Positional arguments for original function
            **kwargs: Keyword arguments for original function

        Returns:
            DataFrame (either new, cached, or as determined by handler)
        """
        logger.debug(f"Intercepting data.get for: {dataset}")

        # 0. Check time travel mode
        tt_context = TimeTravelContext.get_instance()
        if tt_context.is_active():
            return self._handle_time_travel(dataset, tt_context.target_time)

        # 1. Call original data.get
        try:
            new_data = self.original_get(dataset, *args, **kwargs)
        except Exception as e:
            logger.error(f"Original data.get failed for {dataset}: {e}")
            raise

        # Ensure it's a DataFrame
        if not isinstance(new_data, pd.DataFrame):
            # Convert FinlabDataFrame or similar to pandas DataFrame
            new_data = pd.DataFrame(new_data)

        # Keep original data to return to user
        original_data = new_data

        # 2. Apply preprocess hook if registered (for comparison only)
        preprocess_registry = get_preprocess_registry()
        new_data_for_comparison = preprocess_registry.apply(dataset, new_data)

        # 3. Generate backup key
        backup_key = self._generate_backup_key(dataset)

        # 4. Compute hash of preprocessed data
        new_hash = self.hasher.hash_dataframe(new_data_for_comparison)

        # 5. Check for existing backup (stored data is raw/original)
        cached = self.storage.load_latest(backup_key)

        if cached is None:
            # First time - save original data as baseline (not preprocessed)
            logger.info(f"First backup for {dataset}, saving as baseline")
            self.storage.save(backup_key, dataset, original_data, new_hash)
            return original_data

        cached_data, cached_metadata = cached

        # 6. Quick comparison via hash
        if new_hash == cached_metadata.content_hash:
            logger.debug(f"Hash match for {dataset}, data unchanged")
            return original_data

        logger.debug(f"Hash mismatch for {dataset}, performing full comparison")

        # 7. Full comparison - apply preprocess to cached data for comparison
        cached_for_comparison = preprocess_registry.apply(dataset, cached_data)
        result = self.comparer.compare(cached_for_comparison, new_data_for_comparison)

        if result.is_identical:
            # Hash mismatch but identical content (shouldn't happen often)
            logger.debug(f"Full comparison shows identical for {dataset}")
            return original_data

        # 8. Apply policy
        policy = get_policy_for_dataset(
            dataset=dataset,
            default_mode=self.config.comparison.policies.default_mode.value,
            history_modifiable=set(self.config.comparison.policies.history_modifiable),
            threshold=self.config.comparison.change_threshold,
            allow_na_to_value=set(self.config.comparison.policies.allow_na_to_value),
        )

        if not policy.is_violation(result):
            # Changes are within policy - update backup with original data
            logger.info(
                f"Changes accepted for {dataset}: {result.summary()} "
                f"[{policy.name} policy]"
            )
            self.storage.save(backup_key, dataset, original_data, new_hash)
            return original_data

        # 9. Policy violation - create report
        report = AnomalyReport(
            dataset=dataset,
            backup_key=backup_key,
            detected_at=datetime.now(),
            comparison_result=result,
            policy_name=policy.name,
            violation_message=policy.get_violation_message(result),
            old_hash=cached_metadata.content_hash,
            new_hash=new_hash,
        )

        # 10. Save report if configured
        if self.config.anomaly.save_reports:
            report.save(self.config.get_reports_path())

        # 11. Handle anomaly (handler may return cached or new data)
        logger.warning(f"Data anomaly detected: {report.summary}")

        # Both cached_data and original_data are raw/original data
        # Preprocess is only applied during comparison
        return self.handler.handle(report, cached_data, original_data)

    def _generate_backup_key(self, dataset: str) -> str:
        """Generate unique key for backup storage.

        Args:
            dataset: Dataset name

        Returns:
            Sanitized backup key
        """
        # Try to get universe settings from finlab
        universe_hash = self._get_universe_hash()

        return sanitize_backup_key(dataset, universe_hash)

    def _get_universe_hash(self) -> str | None:
        """Get hash of current universe settings.

        Returns:
            Hash string or None if no universe is set
        """
        try:
            # Import finlab to check universe settings
            from finlab import data

            # Try to get current universe settings
            # This depends on finlab's internal API
            if hasattr(data, "_universe") and data._universe is not None:
                universe_str = str(data._universe)
                return hashlib.md5(universe_str.encode()).hexdigest()[:8]

            # Alternative: check for universe context
            if hasattr(data, "universe") and hasattr(data.universe, "_current"):
                universe = data.universe._current
                if universe is not None:
                    universe_str = str(universe)
                    return hashlib.md5(universe_str.encode()).hexdigest()[:8]

        except Exception as e:
            logger.debug(f"Could not get universe hash: {e}")

        return None

    def _handle_time_travel(
        self,
        dataset: str,
        target_time: datetime | None,
    ) -> pd.DataFrame:
        """Handle time travel mode data retrieval.

        Args:
            dataset: Dataset name
            target_time: Target datetime to travel to

        Returns:
            Historical DataFrame from backup

        Raises:
            NoHistoricalDataError: If no backup exists before target time
        """
        if target_time is None:
            msg = "Time travel mode is active but target_time is None"
            raise NoHistoricalDataError(msg)

        backup_key = self._generate_backup_key(dataset)

        # Try to load historical backup
        result = self.storage.load_at_time(backup_key, target_time)

        if result is None:
            target_iso = target_time.isoformat()
            msg = (
                f"No backup found for '{dataset}' at or before {target_iso}. "
                f"Time travel requires historical backups to exist."
            )
            raise NoHistoricalDataError(msg)

        cached_data, metadata = result

        logger.info(
            f"[Time Travel] Loaded backup from {metadata.created_at.isoformat()} "
            f"for '{dataset}' (requested: {target_time.isoformat()})"
        )

        return cached_data


def accept_current_data(
    dataset: str,
    config: SentinelConfig | None = None,
    reason: str | None = None,
) -> bool:
    """Accept current data as new baseline for a dataset.

    This is used to acknowledge and accept anomalous data after review.

    Args:
        dataset: Dataset name to accept
        config: Optional configuration (uses default if not provided)
        reason: Optional reason for accepting

    Returns:
        True if successful, False if dataset not found
    """
    if config is None:
        from finlab_sentinel.config.loader import load_config

        config = load_config()

    storage = ParquetStorage(
        base_path=config.get_storage_path(),
        compression=config.storage.compression,
    )

    # Generate backup key
    backup_key = sanitize_backup_key(dataset)

    # Get latest backup
    cached = storage.load_latest(backup_key)
    if cached is None:
        logger.warning(f"No backup found for {dataset}")
        return False

    # Re-fetch current data
    try:
        from finlab import data as finlab_data

        # Get the original function if we're enabled
        from finlab_sentinel.core.registry import get_original

        original_get = get_original("data.get")
        get_fn = original_get if original_get else finlab_data.get
        new_data = get_fn(dataset)

        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame(new_data)

    except Exception as e:
        logger.error(f"Failed to fetch current data for {dataset}: {e}")
        return False

    # Compute hash and save as accepted
    hasher = ContentHasher()
    new_hash = hasher.hash_dataframe(new_data)

    storage.accept_new_data(
        backup_key=backup_key,
        data=new_data,
        content_hash=new_hash,
        dataset=dataset,
        reason=reason,
    )

    logger.info(
        f"Accepted new data for {dataset}" + (f" (reason: {reason})" if reason else "")
    )

    return True
